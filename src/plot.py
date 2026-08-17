import os

import matplotlib.pyplot as plt
import numpy as np


def fom_display(obj, state, history, ws, i):

    objects = obj if isinstance(obj, list) else [obj]
    every_fom = np.concatenate([item.fom for item in objects]).tolist()
    every_fom.append(np.sum(every_fom))
    every_fom = np.array(every_fom)
    history.all_fom.append(every_fom.copy())
    history.all_beta.append(state.beta)
    with open(os.path.join(ws.save_path, "fom.txt"), 'a') as file:
        every_fom_str = ','.join(map(str, every_fom))
        file.write(f"Iteration {i}\t{every_fom_str}\t{state.beta}\n")
    print(f"fom= {every_fom} beta = {state.beta}")


def fom_training_display(obj, cfg, state, history, ws, i, all_batch_fom):

    every_fom = np.zeros(len(obj.fom_name) + 1)
    every_fom[:-1] = np.sum(all_batch_fom, axis=0) / cfg.dataset.batch_size
    every_fom[-1] = np.sum(all_batch_fom) / cfg.dataset.batch_size

    history.all_fom.append(every_fom.copy())
    history.all_beta.append(state.beta)
    with open(os.path.join(ws.save_path, "fom.txt"), 'a') as file:
        every_fom_str = ','.join(map(str, every_fom))
        file.write(f"Iteration {i}\t{every_fom_str}\t{state.beta}\n")
    print(f"fom= {every_fom} beta = {state.beta}")


def plot_optresult(obj, history, ws, i):
    iterations = range(1, len(history.all_fom) + 1)
    objects = obj if isinstance(obj, list) else [obj]
    foms = np.concatenate([item.fom for item in objects])

    # 误差曲线绘制
    fig, ax1 = plt.subplots()
    ax1.set_yscale('log')
    for j in range(len(foms)):
        ax1.plot(iterations, [sublist[j] for sublist in history.all_fom], label=f'fom {j}')
    ax1.plot(iterations, [sublist[len(foms)] for sublist in history.all_fom],
             label='All fom', color='red')

    # 量化阶段（beta 参数表征）曲线绘制
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.plot(iterations, history.all_beta, label='Beta', color='blue')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fom', color='red')
    ax2.set_ylabel('Beta', color='blue')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(ws.save_path, f'figure_iteration_{i}.png'))
    plt.close(fig)
