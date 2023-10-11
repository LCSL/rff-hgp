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
ns_rffs = np.arange(10, 100, 5)
n_seeds = 50
for experiment in ['proof-of-concept', 'assembly-task', 'bed-making']:
    # Load exact GP
    gp_type = 'exact'
    seed = 0
    n_rffs = 0

    with open(f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_train_rff_GP_True_n_rffs_{n_rffs}.txt",
            "rb") as fp:
        reference_signals = pickle.load(fp)

    with open(f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_train_rff_GP_True_n_rffs_{n_rffs}.txt",
            "rb") as fp:
        uncertainties = pickle.load(fp)

    exact_mu = np.array(reference_signals)
    exact_var = np.array(uncertainties)
    plt.figure(figsize=[7, 3.2])
    line_styles = ['-', '--', '-.']

    for indx_approx, gp_type in enumerate(['rffs', 'nystrom']):
        median_mu_errors = []
        lowperc_mu_errors = []
        uppperc_mu_errors = []

        for n_rffs in ns_rffs:
            print(n_rffs)
            curr_opti_times = []
            curr_pred_times = []
            curr_mu_errors = []
            curr_var_errors = []
            for seed in range(0, n_seeds):
                if seed % 10 == 0:
                    print(seed)
                with open(f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_incremental_n_feat_{n_rffs}.txt",
                        "rb") as fp:
                    reference_signals = pickle.load(fp)
                col = 'C3' if gp_type == 'rffs' else 'C4'
                curr_mu_errors.append(np.sqrt(np.mean(np.sum(np.square(reference_signals - exact_mu), axis=1) / np.sum(np.square(exact_mu), axis=1))) * 100)

            median_mu_errors.append(np.median(curr_mu_errors))
            lowperc_mu_errors.append(np.percentile(curr_mu_errors, q=15))
            uppperc_mu_errors.append(np.percentile(curr_mu_errors, q=85))

        median_mu_errors = np.array(median_mu_errors).squeeze()
        lowperc_mu_errors = np.array(lowperc_mu_errors).squeeze()
        uppperc_mu_errors = np.array(uppperc_mu_errors).squeeze()
        plt.plot(ns_rffs.squeeze(), median_mu_errors, linewidth=2, linestyle=line_styles[indx_approx])
        plt.fill_between(ns_rffs.squeeze(), lowperc_mu_errors, uppperc_mu_errors, alpha=0.3, label='_nolegend_')
    # exit(0)
    plt.legend([r"RF-HGP", r"Nystr\"om-HGP"], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', borderaxespad=0, ncol=2)
    plt.xlabel("Number of features")
    plt.ylabel("$\mathrm{RMSE}_\%$ mean")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{experiment}_mu_errors_incremental.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure(figsize=[7, 3.2])
    for indx_approx, gp_type in enumerate(['rffs', 'nystrom']):
        median_var_errors = []
        lowperc_var_errors = []
        uppperc_var_errors = []

        for n_rffs in ns_rffs:
            print(n_rffs)
            curr_opti_times = []
            curr_pred_times = []
            curr_mu_errors = []
            curr_var_errors = []
            for seed in range(0, n_seeds):
                if seed % 10 == 0:
                    print(seed)

                with open(f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_incremental_n_feat_{n_rffs}.txt",
                        "rb") as fp:
                    uncertainties = pickle.load(fp)
                col = 'C5' if gp_type == 'rffs' else 'C6'
                curr_var = np.array(uncertainties).squeeze()
                curr_var_errors.append(np.sqrt(np.mean(np.sum(np.square(curr_var - exact_var), axis=1) / np.sum(np.square(exact_var), axis=1))) * 100)

            median_var_errors.append(np.median(curr_var_errors))
            lowperc_var_errors.append(np.percentile(curr_var_errors, q=15))
            uppperc_var_errors.append(np.percentile(curr_var_errors, q=85))

        median_var_errors = np.array(median_var_errors).squeeze()
        lowperc_var_errors = np.array(lowperc_var_errors).squeeze()
        uppperc_var_errors = np.array(uppperc_var_errors).squeeze()

        plt.plot(ns_rffs.squeeze(), median_var_errors, linewidth=2, linestyle=line_styles[indx_approx])
        plt.fill_between(ns_rffs.squeeze(), lowperc_var_errors, uppperc_var_errors, alpha=0.3, label='_nolegend_')

    plt.legend(["RF-HGP", r"Nystr\"om-HGP"], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand',
               borderaxespad=0, ncol=2)

    plt.xlabel("Number of features")
    plt.ylabel("$\mathrm{RMSE}_{\%}$ var.")
    plt.grid('True', which='both')
    plt.yscale('log')
    plt.tight_layout()

    plt.savefig(f"figures/{experiment}_var_errors_incremental.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()





