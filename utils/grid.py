import subprocess

params = [
    {
        'scaling_factor' : scaling_factor,
        'is_simple' : is_simple,
        'l_mse' : l_mse,
        'l_adv' : l_adv,
        'l_vgg' : l_vgg,
    }
    # for scaling_factor in [2, 4]
    for scaling_factor in [2]
    for is_simple in [True, False]
    for l_mse in [0.0, 0.3]
    for l_vgg in [0.0, 0.01]
    # for l_vgg in [0.0]
    for l_adv in [0.0, 0.01]
]

for p in params:
    scaling_factor = p['scaling_factor']
    is_simple = p['is_simple']
    l_mse = p['l_mse']
    l_vgg = p['l_vgg']
    l_adv = p['l_adv']

    coma = ['python', '-m', 'scripts.train']
    coma.extend([f'--scaling_factor={scaling_factor}'])
    coma.extend([f'--l_mse={l_mse}'])
    coma.extend([f'--l_adv={l_adv}'])
    coma.extend([f'--l_vgg={l_vgg}'])
    if is_simple:
        coma.extend([f'--is_simple'])

    subprocess.run(coma)
