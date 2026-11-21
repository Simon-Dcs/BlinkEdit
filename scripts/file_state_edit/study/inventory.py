"""Check each required data item, by its expected filename, not merely total counts."""

from prepare import R, read, save


def inspect():
    m = read(R / 'manifest.json')
    root = R / 'setup/data'
    rows = []
    for key in m['formal']['edithero']:
        n = m['chain_info'][key]['n_turns']
        paths = [root / 'part_masks' / key / 'masks.json']
        for i in range(n + 1):
            t = f'turn{i:02}'
            paths.append(root / 'states' / key / (t + '.glb'))
            for v in range(4):
                img = f"view{v}_{'heldout' if v else 'cond'}.png"
                paths.append(root / 'renders' / key / t / img)
                if i:
                    paths.append(
                        root / 'part_masks' / key / t / ('edit_' + img)
                    )
        missing = [
            str(p.relative_to(root))
            for p in paths
            if not p.is_file() or not p.stat().st_size
        ]
        rows.append(
            dict(
                chain=key,
                expected=len(paths),
                present=len(paths) - len(missing),
                missing=missing,
            )
        )
    return rows


if __name__ == '__main__':
    rows = inspect()
    save(R / 'metrics/data-inventory.json', rows)
    print(f"{sum(not x['missing'] for x in rows)}/{len(rows)} full chains")
