'''
Author: Edoardo Caldarelli
Affiliation: Institut de Robòtica i Informàtica Industrial, CSIC-UPC
email: ecaldarelli@iri.upc.edu
October 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 22})

train_rff_GP = False
n_dofs = 6

ns_rffs = np.arange(10, 250, 10)
n_seeds = 50

plt.figure(figsize=[8, 3.4])
experiments_to_be_rerun = []
color_cycle = ['C2', 'C9', 'C3']
style_cycle = ['-', '--', '-.']

ns_samples = [1286, 1222, 864]
nus = [0.62853, 0.63603, 0.35879]
for exp_indx, experiment in enumerate(['proof-of-concept', 'assembly-task', 'bed-making']):
    # Load exact GP
    n_samples = ns_samples[exp_indx]
    nu = nus[exp_indx]
    gp_type = 'exact'
    seed = 0
    n_rffs = 0
    with open(f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_train_rff_GP_False_n_rffs_{n_rffs}.txt",
              "rb") as fp:
        reference_signals = pickle.load(fp)

    exact_mu = np.array(reference_signals)

    gp_type = 'rffs'

    median_mu_errors = []
    lowperc_mu_errors = []
    uppperc_mu_errors = []
    exact_mu = exact_mu.squeeze()

    # Plot Error Means
    for n_rffs in ns_rffs:
            print(n_rffs)
            curr_opti_times = []
            curr_pred_times = []
            curr_mu_errors = []
            curr_var_errors = []

            for seed in range(0, n_seeds):
                if(seed % 10 == 0):
                    print(seed)
                with open(f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                        "rb") as fp:
                    reference_signals = pickle.load(fp)
                reference_signals_array = np.empty((0, 2000, 1))
                for r in reference_signals:
                    reference_signals_array = np.concatenate((reference_signals_array, np.expand_dims(r, axis=0)), axis=0)
                indices = []
                try:
                    indices = np.argwhere(np.isnan(reference_signals_array))
                except:
                    pass
                reference_signals = np.array(reference_signals).squeeze()
                curr_mu_errors.append(np.sqrt(np.mean(np.sum(np.square(reference_signals - exact_mu), axis=1) / np.sum(np.square(exact_mu), axis=1))) * 100)
                if len(indices) != 0:
                    print("Found NaN elements at indices ", indices, " for seed ", seed, " and n rffs ", n_rffs, " experiment ", experiment)
                    experiments_to_be_rerun.append((experiment, seed, n_rffs))

            median_mu_errors.append(np.median(curr_mu_errors))
            lowperc_mu_errors.append(np.percentile(curr_mu_errors, q=15))
            uppperc_mu_errors.append(np.percentile(curr_mu_errors, q=85))

    median_mu_errors = np.array(median_mu_errors).squeeze()
    lowperc_mu_errors = np.array(lowperc_mu_errors).squeeze()
    uppperc_mu_errors = np.array(uppperc_mu_errors).squeeze()

    plt.plot(ns_rffs.squeeze(), median_mu_errors, linewidth=2, color=color_cycle[exp_indx], linestyle=style_cycle[exp_indx])
    plt.fill_between(ns_rffs.squeeze(), lowperc_mu_errors, uppperc_mu_errors, alpha=0.3, label='_nolegend_', color=color_cycle[exp_indx])
    plt.plot(ns_rffs, nu * n_samples * np.sqrt(np.log(n_samples)) * np.reciprocal(np.sqrt(ns_rffs)), label='_nolegend_', color='C4', linewidth=3)

plt.ylabel(r'$\mathrm{RMSE}_{\%}$ mean')
plt.xlabel('Number of RFFs')
plt.yscale('log')
plt.grid(visible=True, which='both')
plt.legend(["Free motion", "Assembly", "Bed-making"], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand',
               borderaxespad=0, ncol=3, handlelength=1.0)
plt.tight_layout()
plt.savefig(f"figures/rate_all_mu_errors_train_rff_GP_{train_rff_GP}.png", dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
plt.figure(figsize=[8, 3.4])

print(experiments_to_be_rerun)
for exp_indx, experiment in enumerate(['proof-of-concept']): #, 'assembly-task', 'bed-making']):
    # Load exact GP
    n_samples = ns_samples[exp_indx]
    nu = nus[exp_indx]

    gp_type = 'exact'
    seed = 0
    n_rffs = 0
    with open(f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_train_rff_GP_False_n_rffs_{n_rffs}.txt",
              "rb") as fp:
        uncertainties = pickle.load(fp)

    exact_var = np.array(uncertainties)

    gp_type = 'rffs'

    median_var_errors = []
    lowperc_var_errors = []
    uppperc_var_errors = []

    # Plot Error Variance
    for n_rffs in ns_rffs:
        print(n_rffs)
        curr_opti_times = []
        curr_pred_times = []
        curr_mu_errors = []
        curr_var_errors = []
        for seed in range(0, n_seeds):
            if seed % 10 == 0:
                print(seed)
            with open(
                    f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                    "rb") as fp:
                uncertainties = pickle.load(fp)
            curr_var = np.array(uncertainties).squeeze()
            curr_var_errors.append(np.sqrt(np.mean(np.sum(np.square(curr_var - exact_var), axis=1) / np.sum(np.square(exact_var), axis=1))) * 100)

        median_var_errors.append(np.median(curr_var_errors))
        lowperc_var_errors.append(np.percentile(curr_var_errors, q=15))
        uppperc_var_errors.append(np.percentile(curr_var_errors, q=85))

    median_var_errors = np.array(median_var_errors).squeeze()
    lowperc_var_errors = np.array(lowperc_var_errors).squeeze()
    uppperc_var_errors = np.array(uppperc_var_errors).squeeze()
    plt.plot(ns_rffs.squeeze(), median_var_errors, linewidth=2, color=color_cycle[exp_indx], linestyle=style_cycle[exp_indx])
    plt.fill_between(ns_rffs.squeeze(), lowperc_var_errors, uppperc_var_errors, alpha=0.3, label='_nolegend_', color=color_cycle[exp_indx])
    plt.plot(ns_rffs, nu * n_samples * np.sqrt(np.log(n_samples)) * np.reciprocal(np.sqrt(ns_rffs)), label='_nolegend_', color='C4', linewidth=3)

plt.ylabel(r'$\mathrm{RMSE}_{\%}$ var.')
plt.xlabel('Number of RFFs')
plt.yscale('log')
plt.legend(["Free motion", "Assembly", "Bed-making"], bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), loc='lower left',
           mode='expand',
           borderaxespad=0, ncol=3, handlelength=1.0)

plt.grid(visible=True, which='both')
plt.tight_layout()
plt.savefig(f"figures/rate_all_var_errors_train_rff_GP_{train_rff_GP}.png", dpi=300, bbox_inches='tight',
            pad_inches=0)

plt.show()







