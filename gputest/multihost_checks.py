"""Plain multi-host checks, split out because checks.py is near 600 lines."""

from __future__ import annotations

import ipaddress
import json
import shlex
import statistics
from collections import defaultdict
from dataclasses import replace

from gputest import parsers
from gputest.core import Result, RunContext, check


def _parallel(ctx, cmd, nodes, fixture):
    return ctx.run_parallel([{'cmd': cmd, 'node': n, 'timeout': ctx.config['executor']['timeout_s'],
                                  'mock_fixture': fixture} for n in nodes])


def _inventory(ctx, nodes=None):
    cfg = ctx.config['network']
    cmd = [ctx.config['tools']['curl'], '-fsSL', '--max-time', str(cfg['imds_timeout_s']),
           '-H', 'Authorization: Bearer Oracle', cfg['imds_url']]
    nodes = (ctx.nodes or [None]) if nodes is None else nodes
    results = _parallel(ctx, cmd, nodes, 'imds_host.json')
    inventory = {}
    for node, rr in zip(nodes, results):
        override = ctx.config['placement']['inventory'].get(node)
        if rr.skipped:
            inventory[node] = {'gmf': '<GMF-from-IMDS>', 'local_block': '<block-from-IMDS>', 'planes': None, 'dry_run': True}
            continue
        try:
            data = json.loads(rr.stdout) if rr.returncode == 0 else {}
            topology = data.get('rdmaTopologyData', {})
            inventory[node] = {'gmf': topology.get('customerGpuMemoryFabric'),
                               'local_block': topology.get('customerLocalBlock'),
                               'planes': data.get('rdmaFabricData', {}).get('planes', 1) if data else None}
        except (ValueError, AttributeError):
            inventory[node] = {}
        if override is not None:
            inventory[node].update(override)
    return inventory


def select_nodes(inventory, count, rack_max):
    """Select one GMF for small jobs, prefer a common local block for larger jobs."""
    if count < 2 or count > len(inventory):
        raise ValueError('node count must be between 2 and the inventory size')
    if any(not entry.get('gmf') for entry in inventory.values()):
        raise ValueError('GMF metadata unavailable; configure placement.inventory')
    gmfs, blocks = defaultdict(list), defaultdict(list)
    for host, entry in inventory.items():
        gmfs[entry['gmf']].append(host)
        if entry.get('local_block'):
            blocks[entry['local_block']].append(host)
    groups = gmfs if count <= rack_max else blocks
    candidates = [hosts for hosts in groups.values() if len(hosts) >= count]
    if candidates:
        return min(candidates, key=lambda hosts: (-len(hosts), hosts))[:count]
    if count <= rack_max:
        raise ValueError(f'no single GMF contains {count} hosts; reduce --node-count')
    return [host for hosts in gmfs.values() for host in hosts][:count]


def _devices(ctx, nodes):
    results = _parallel(ctx, [ctx.config['tools']['rdma'], 'link'], nodes,
                        f'rdma_link_{ctx.network_mode}.txt')
    cfg = ctx.network_config()
    pins = cfg.get('vf_rails', []) if ctx.network_mode == 'multiplanar' else cfg.get('hca_names', [])
    found = {}
    for node, rr in zip(nodes, results):
        if rr.skipped:
            # Placeholders are explicit: dry-run performs no remote discovery.
            names = pins or [f'<discovered-device-{i}>'
                             for i in range(cfg.get('expected_active_hcas', ctx.sku_config()['gpu_count']))]
            found[node] = [{'hca': n, 'netdev': f'<netdev-for-{n}>'} for n in names]
            continue
        if rr.returncode:
            raise ValueError(f'{node}: RDMA discovery failed: {rr.stderr}')
        active = parsers.parse_rdma_link(rr.stdout)
        active = [d for d in active if d['hca'] != cfg.get('frontend_hca')]
        by_name = {d['hca']: d for d in active}
        names = pins or list(by_name)
        if not names or any(n not in by_name for n in names):
            raise ValueError(f'{node}: required RDMA devices are not ACTIVE + LINK_UP')
        expected = len(pins) if pins else cfg.get('expected_active_hcas', ctx.sku_config()['gpu_count'])
        if len(names) != expected:
            raise ValueError(f'{node}: found {len(names)} devices, expected {expected}')
        found[node] = [by_name[n] for n in names]
        if any(not d['netdev'] for d in found[node]):
            raise ValueError(f'{node}: RDMA netdev mapping missing')
    return found


@check('rdma_iface', category='preflight')
def rdma_iface(ctx: RunContext) -> Result:
    nodes = ctx.nodes or [None]
    devices = _devices(ctx, nodes)
    raw = []
    if ctx.network_mode == 'multiplanar':
        for node in nodes:
            rr = ctx.run([ctx.config['tools']['ip'], '-j', '-d', 'link', 'show'], node=node,
                         timeout=ctx.config['executor']['timeout_s'], mock_fixture='vrf_links.json')
            raw.append(rr.stdout + rr.stderr)
            if rr.skipped:
                continue
            if rr.returncode:
                raise ValueError(f'{node}: VRF discovery failed')
            links = json.loads(rr.stdout)
            by_name = {link['ifname']: link for link in links}
            by_index = {link['ifindex']: link['ifname'] for link in links}
            for index, device in enumerate(devices[node]):
                vrf = ctx.network_config()['vrfs'][index]
                if vrf not in by_name or by_name[vrf].get('linkinfo', {}).get('info_kind') != 'vrf':
                    raise ValueError(f'{node}: missing VRF {vrf}')
                master = by_name.get(device['netdev'], {}).get('master')
                if by_index.get(master, master) != vrf:
                    raise ValueError(f'{node}: unbound VF {device["hca"]}, expected master {vrf}')
    return Result('rdma_iface', 'SKIP' if ctx.dry_run else 'PASS',
                  'dry-run: discovery placeholders' if ctx.dry_run else 'RDMA devices discovered',
                  {'devices': devices}, _raw='\n'.join(raw))


def _address(ctx, node, device, family):
    rr = ctx.run([ctx.config['tools']['ip'], '-j', 'address', 'show', 'dev', device['netdev']],
                 node=node, timeout=ctx.config['executor']['timeout_s'],
                 mock_fixture='rdma_address.json')
    if rr.skipped:
        return f'<{family}-address-of-{node}-{device["hca"]}>'
    if rr.returncode:
        raise ValueError(f'{node}: address discovery failed: {rr.stderr}')
    addresses = parsers.parse_ip_addresses(rr.stdout).get(device['netdev'], [])
    for af in (('inet', 'inet6') if family == 'auto' else (family,)):
        for a in addresses:
            ip = ipaddress.ip_address(a['local'])
            if a['family'] == af and not (ip.is_loopback or ip.is_link_local or ip.is_unspecified):
                return str(ip)
    raise ValueError(f'{node}: no routable {family} address on {device["netdev"]}')


def _rdma(ctx, name, kinds):
    nodes = ctx.nodes[:2]
    iface = rdma_iface(replace(ctx, nodes=nodes))
    devices = {n: iface.metrics['devices'][n] for n in nodes}
    server, client = nodes
    if len(devices[server]) != len(devices[client]):
        raise ValueError('server/client RDMA device counts differ')
    cfg = ctx.config['rdma_perf']
    mode = cfg[ctx.network_mode]
    rails, raw, issues, uncalibrated = [], [], [], False
    for index, (sd, cd) in enumerate(zip(devices[server], devices[client])):
        family = mode['ip_family']
        address = _address(ctx, server, sd, family)
        # Verify the client can use the same family, without guessing a VRF.
        actual_family = 'inet6' if ':' in address else ('inet' if not ctx.dry_run else family)
        _address(ctx, client, cd, actual_family)
        rail = {'server_device': sd['hca'], 'client_device': cd['hca']}
        prefix = []
        if ctx.network_mode == 'multiplanar':
            net = ctx.network_config()
            prefix = [ctx.config['tools']['ip'], 'vrf', 'exec', net['vrfs'][index],
                      ctx.config['tools']['numactl'], '-N', str(net['numa_map'][str(index)])]
        for kind in kinds:
            port = cfg['port'] + index
            extra = list(mode['extra_args'])
            if ctx.network_mode == 'single_plane' and actual_family == 'inet6':
                extra += ['--ipv6', '--ipv6-addr']
            args = cfg[f'{kind}_args'] + extra + ['-p', str(port)]
            tool = ctx.config['tools'][f'ib_write_{kind}']
            jobs = [{'cmd': prefix + [tool] + args + ['-d', d['hca']] + peer,
                         'node': node, 'timeout': cfg['timeout_s'], 'mock_fixture': f'ib_write_{kind}.txt'}
                    for node, d, peer in ((server, sd, []), (client, cd, [address]))]
            sr, cr = ctx.run_pair(*jobs, port)
            raw += [f'{node} {kind}\n{r.stdout}\n{r.stderr}' for node, r in ((server, sr), (client, cr))]
            if ctx.dry_run:
                continue
            parsed = parsers.parse_perftest(cr.stdout, kind)
            value = parsed['value']
            metric = 'bw_gbps' if kind == 'bw' else 'lat_us'
            rail[metric] = value
            if sr.returncode or cr.returncode or value is None:
                issues.append(f'{sd["hca"]} {kind}: command failed or no result')
                continue
            limit = mode['bw_min_gbps' if kind == 'bw' else 'lat_max_us']
            if limit is None:
                uncalibrated = True
            elif (value < limit if kind == 'bw' else value > limit):
                issues.append(f'{sd["hca"]} {metric}={value}, limit={limit}')
        rails.append(rail)
    metrics = {'nodes': nodes, 'rails': rails, 'threshold_source': mode.get('threshold_source', 'config default'),
               'uncalibrated': uncalibrated}
    for metric in ('bw_gbps', 'lat_us'):
        values = [r[metric] for r in rails if r.get(metric) is not None]
        if values:
            metrics[metric + '_spread_pct'] = (max(values) - min(values)) / statistics.mean(values) * 100
    status = 'SKIP' if ctx.dry_run else ('FAIL' if issues else ('WARN' if uncalibrated else 'PASS'))
    result = Result(name, status, 'dry-run: address/device placeholders' if ctx.dry_run else
                    ('; '.join(issues) or f'{len(rails)} rails; ' +
                     ('thresholds unset' if uncalibrated else 'within configured thresholds')),
                    metrics, _raw='\n'.join(raw))
    return result


@check('rdma_perf', category='rdma', requires_multi_host=True)
def rdma_perf(ctx: RunContext) -> Result:
    return _rdma(ctx, 'rdma_perf', ['bw', 'lat'])


@check('rdma_bw', category='rdma', requires_multi_host=True)
def rdma_bw(ctx: RunContext) -> Result:
    return _rdma(ctx, 'rdma_bw', ['bw'])


@check('rdma_lat', category='rdma', requires_multi_host=True)
def rdma_lat(ctx: RunContext) -> Result:
    return _rdma(ctx, 'rdma_lat', ['lat'])


def _nccl(ctx, collective):
    cfg = ctx.config['nccl']
    mode = cfg[ctx.network_mode]
    count = ctx.node_count if ctx.node_count is not None else len(ctx.nodes)
    inventory = _inventory(ctx)
    nodes = select_nodes(inventory, count, ctx.config['placement']['rack_max_nodes'])
    devices = rdma_iface(replace(ctx, nodes=nodes)).metrics['devices']
    env = {k: str(v) for k, v in mode['env'].items()}
    if ctx.network_mode == 'multiplanar':
        group = 'small_cluster' if count <= mode['small_max_nodes'] else 'large_cluster'
        env.update({k: str(v) for k, v in mode[group].items()})
    else:
        names = [[d['hca'] for d in devices[n]] for n in nodes]
        if any(set(n) != set(names[0]) for n in names):
            raise ValueError('NCCL requires matching discovered HCA names on all hosts')
        env['NCCL_IB_HCA'] = ','.join(names[0])
    if env.get('NCCL_MNNVL_ENABLE') == '1' and ctx.sku_config().get('imex_capable'):
        imex_result = imex(ctx, nodes, inventory)
        if imex_result.status == 'FAIL':
            return Result(f'nccl_{collective}', 'FAIL', imex_result.message,
                          _raw=imex_result._raw)
    ranks = ctx.sku_config()['gpu_count']
    command = [ctx.config['tools']['mpirun'], '-np', str(count * ranks), '--host',
               ','.join(f'{n}:{ranks}' for n in nodes), '--map-by', f'ppr:{ranks}:node']
    command += mode['mpi_args']
    for key, value in env.items():
        command += ['-x', f'{key}={value}']
    command += [ctx.config['tools'][f'{collective}_perf']] + cfg['args']
    rr = ctx.run(command, node=nodes[0], env=env, timeout=cfg['timeout_s'],
                 mock_fixture=f'nccl_{collective}.txt')
    parsed = parsers.parse_nccl(rr.stdout)
    largest = parsed['largest']
    limit = mode['thresholds'][collective]
    source = 'configured threshold'
    reference = mode['example_reference_gbs'][collective].get(count)
    if limit is None and reference is not None:
        limit = reference * cfg['baseline_threshold_pct'] / 100
        source = 'example reference data (not a hardware guarantee)'
    issues = []
    blocks = {inventory[n].get('local_block') for n in nodes}
    placement_warning = (count > ctx.config['placement']['rack_max_nodes']
                         and (None in blocks or len(blocks) > 1))
    values = {}
    if not ctx.dry_run:
        if rr.returncode or not largest:
            issues.append('launcher failed or result table missing')
        else:
            if largest['size'] != cfg['largest_size_bytes']:
                issues.append('largest requested message size not completed')
            if any(row[label]['wrong'] not in (None, 0) for row in parsed['rows']
                   for label in ('in_place', 'out_of_place')):
                issues.append('NCCL reported data validation errors')
            values = {label: largest[label]['busbw'] for label in ('in_place', 'out_of_place')}
            if min(values.values()) <= 0:
                issues.append('non-positive bus bandwidth')
            if limit is not None and min(values.values()) < limit:
                issues.append(f'largest-size busbw below {limit} GB/s')
    groups = defaultdict(list)
    for n in nodes:
        groups[inventory[n]['gmf']].append(n)
    metrics = {'nodes': nodes, 'gmf_groups': dict(groups), 'ranks': count * ranks,
               'env': env, **parsed, 'busbw_min_gbs': limit, 'threshold_source': source,
               'command': shlex.join(command), 'uncalibrated': limit is None and not issues and not placement_warning,
               'placement_warning': placement_warning}
    status = 'SKIP' if ctx.dry_run else ('FAIL' if issues else ('WARN' if limit is None or placement_warning else 'PASS'))
    result = Result(f'nccl_{collective}', status,
                    'dry-run: placement/device placeholders' if ctx.dry_run else
                    ('; '.join(issues) or ('threshold unset; calibrate baseline' if limit is None else source)),
                    metrics, _raw=rr.stdout + rr.stderr)
    if placement_warning and not ctx.dry_run:
        result.message += '; mismatched or unknown customerLocalBlock across racks'
    return result


@check('nccl_allreduce', category='nccl', requires_multi_host=True)
def nccl_allreduce(ctx: RunContext) -> Result:
    return _nccl(ctx, 'allreduce')


@check('nccl_alltoall', category='nccl', requires_multi_host=True)
def nccl_alltoall(ctx: RunContext) -> Result:
    return _nccl(ctx, 'alltoall')


@check('imex', category='preflight', skus=['gb300'], requires_multi_host=True)
def imex(ctx: RunContext, nodes=None, inventory=None) -> Result:
    """Validate existing per-GMF membership; never configure/restart across racks.

    https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html
    """
    nodes = ctx.nodes if nodes is None else nodes
    inventory = _inventory(ctx) if inventory is None else inventory
    if any(not inventory[n].get('gmf') for n in nodes):
        return Result('imex', 'FAIL', 'GMF metadata required to validate IMEX membership')
    cfg = ctx.config['imex']
    files = _parallel(ctx, [ctx.config['tools']['cat'], cfg['nodes_file']], nodes, 'imex_nodes.txt')
    services = _parallel(ctx, [ctx.config['tools']['systemctl'], 'is-active', cfg['service']],
                         nodes, 'imex_service.txt')
    issues, members, raw = [], {}, []
    for node, file, service in zip(nodes, files, services):
        raw += [f'{node}\n{file.stdout}\n{file.stderr}\n{service.stdout}\n{service.stderr}']
        if ctx.dry_run:
            continue
        listed = [line.split('#')[0].strip() for line in file.stdout.splitlines()]
        listed = sorted(n for n in listed if n)
        expected = sorted(n for n in nodes if inventory[n]['gmf'] == inventory[node]['gmf'])
        # Exact inventory names required; aliases/IPs must be listed in --nodes too.
        known = {**ctx.config['placement']['inventory'], **inventory}
        if (file.returncode or len(listed) > ctx.config['placement']['rack_max_nodes']
                or len(listed) != len(set(listed)) or not set(expected).issubset(listed)
                or any(n not in known or known[n].get('gmf') != inventory[node]['gmf'] for n in listed)):
            issues.append(f'{node}: IMEX membership must contain selected peers and only its GMF')
        if service.returncode or service.stdout.strip() != 'active':
            issues.append(f'{node}: IMEX service inactive')
        members[node] = listed
    return Result('imex', 'SKIP' if ctx.dry_run else ('FAIL' if issues else 'PASS'),
                  'dry-run' if ctx.dry_run else ('; '.join(issues) or 'IMEX membership isolated per GMF'),
                  {'members': members}, _raw='\n'.join(raw))


@check('network_mode', category='preflight')
def network_mode(ctx: RunContext) -> Result:
    inventory = _inventory(ctx)
    expected = ctx.config['network']['expected_planes'][ctx.network_mode]
    issues, unavailable = [], []
    for node, entry in inventory.items():
        planes = entry.get('planes')
        if planes is None:
            unavailable.append(node or 'localhost')
        elif planes != expected:
            issues.append(f'{node or "localhost"}: wrong plane count {planes}; expected {expected}')
    return Result('network_mode', 'SKIP' if ctx.dry_run else ('FAIL' if issues else ('WARN' if unavailable else 'PASS')),
                  'dry-run' if ctx.dry_run else ('; '.join(issues) or
                  (f'IMDS unavailable on {unavailable}; using {ctx.network_mode}' if unavailable else ctx.network_mode)),
                  {'hosts': {str(n): v for n, v in inventory.items()}})


@check('placement', category='preflight', requires_multi_host=True)
def placement(ctx: RunContext) -> Result:
    inventory = _inventory(ctx)
    count = ctx.node_count if ctx.node_count is not None else len(ctx.nodes)
    nodes = select_nodes(inventory, count, ctx.config['placement']['rack_max_nodes'])
    blocks = {inventory[n].get('local_block') for n in nodes}
    warning = count > ctx.config['placement']['rack_max_nodes'] and (None in blocks or len(blocks) > 1)
    return Result('placement', 'SKIP' if ctx.dry_run else ('WARN' if warning else 'PASS'),
                  'dry-run' if ctx.dry_run else ('mismatched or unknown customerLocalBlock across racks' if warning
                  else 'GMF/local-block placement selected'),
                  {'nodes': nodes, 'inventory': inventory})
