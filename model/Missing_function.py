
######Missing########
def apply_missing_fixed(a, v, p_missing=0.8, device='cuda', mode='alternate', return_mask=False):

    batch_size = a.size(0)
    num_missing = int(batch_size * p_missing)
    missing_mask = torch.zeros(batch_size, 2, device=device)

    if mode == 'fundus':
        missing_mask[:num_missing, 0] = 1
    elif mode == 'oct':
        missing_mask[:num_missing, 1] = 1
    elif mode == 'alternate':
        for idx in range(num_missing):
            if idx % 2 == 0:
                missing_mask[idx, 0] = 1
            else:
                missing_mask[idx, 1] = 1
    else:
        raise ValueError("Invalid mode. Choose from ['fundus', 'oct', 'alternate'].")

    # print(missing_mask)
    a_new, v_new = a.clone(), v.clone()
    for i in range(batch_size):
        if missing_mask[i, 0] == 1:
            a_new[i] = torch.zeros_like(a_new[i])
        if missing_mask[i, 1] == 1:
            v_new[i] = torch.zeros_like(v_new[i])

    if return_mask:
        mask_a, mask_v = 1 - missing_mask[:, 0], 1 - missing_mask[:, 1]
        return a_new, v_new, missing_mask, mask_a, mask_v
    else:
        return a_new, v_new
