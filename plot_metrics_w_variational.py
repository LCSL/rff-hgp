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

train_rff_GP = True
n_dofs = 6
ns_rffs = np.arange(10, 200, 40)
n_seeds = 5
experiments = ['proof-of-concept', 'assembly-task', 'bed-making']
gp_types = ['rffs', 'svgp']

for experiment in experiments:
    # Load exact GP
    gp_type = 'exact'
    seed = 0
    n_rffs = 0

    with open(f"{experiment}/pred_times_{gp_type}_seed_{seed}_train_rff_GP_True_n_rffs_{n_rffs}.txt",
              "rb") as fp:
        prediction_times = pickle.load(fp)

    with open(f"{experiment}/opti_times_gp_{gp_type}_seed_{seed}_train_rff_GP_True_n_rffs_{n_rffs}.txt",
              "rb") as fp:
        optimization_times = pickle.load(fp)

    exact_pred_time_mean = np.array(np.median(prediction_times)).squeeze()
    exact_lowperc_pred_time = np.array(np.percentile(prediction_times, q=15)).squeeze()
    exact_uppperc_pred_time = np.array(np.percentile(prediction_times, q=85)).squeeze()

    exact_opti_time_mean = np.array(np.median(optimization_times)).squeeze()
    exact_lowperc_opti_time = np.array(np.percentile(optimization_times, q=15)).squeeze()
    exact_uppperc_opti_time = np.array(np.percentile(optimization_times, q=85)).squeeze()

    plt.figure(figsize=[8, 3])

    for gp_type in gp_types:
        median_pred_times = []
        lowperc_pred_times = []
        uppperc_pred_times = []
        # Plot times
        for n_rffs in ns_rffs:
            print(n_rffs)
            curr_opti_times = []
            curr_pred_times = []
            curr_mu_errors = []
            curr_var_errors = []
            for seed in range(0, n_seeds):
                if (seed % 10 == 0):
                    print(seed)

                with open(f"{experiment}/pred_times_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                          "rb") as fp:
                    prediction_times = pickle.load(fp)

                with open(
                        f"{experiment}/opti_times_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                        "rb") as fp:
                    optimization_times = pickle.load(fp)

                curr_pred_times.append(prediction_times)
                curr_opti_times.append(optimization_times)

            median_pred_times.append(np.median(curr_pred_times))
            lowperc_pred_times.append(np.percentile(curr_pred_times, q=15))
            uppperc_pred_times.append(np.percentile(curr_pred_times, q=85))

        median_pred_times = np.array(median_pred_times).squeeze()
        lowperc_pred_times = np.array(lowperc_pred_times).squeeze()
        uppperc_pred_times = np.array(uppperc_pred_times).squeeze()
        plt.plot(ns_rffs.squeeze(), median_pred_times, linewidth=2)
        plt.fill_between(ns_rffs.squeeze(), lowperc_pred_times, uppperc_pred_times, alpha=0.3, label='_nolegend_')

    plt.plot(ns_rffs.squeeze(), np.repeat(exact_pred_time_mean, ns_rffs.shape), linewidth=2, linestyle='--')
    plt.fill_between(ns_rffs.squeeze(), exact_lowperc_pred_time, exact_uppperc_pred_time, alpha=0.3, label='_nolegend_')
    plt.ylabel('Pred.\ time [s]')
    plt.xlabel('Number of RFFs')
    plt.grid(True)
    plt.legend(['RF-HGP', 'SVGP-HGP', 'Exact HGP'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand',
               borderaxespad=0, ncol=3, handlelength=1)
    plt.tight_layout()
    plt.savefig(f"figures/new_{experiment}_prediction_times_train_rff_GP_{train_rff_GP}.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()
    plt.figure(figsize=[8, 3])
    if train_rff_GP:
        for type_indx, gp_type in enumerate(gp_types):

            median_opti_times = []
            lowperc_opti_times = []
            uppperc_opti_times = []

            # Plot times
            for n_rffs in ns_rffs:
                print(n_rffs)
                curr_opti_times = []
                curr_pred_times = []
                curr_mu_errors = []
                curr_var_errors = []
                for seed in range(0, n_seeds):
                    if (seed % 10 == 0):
                        print(seed)

                    with open(
                            f"{experiment}/pred_times_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                            "rb") as fp:
                        prediction_times = pickle.load(fp)

                    with open(
                            f"{experiment}/opti_times_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                            "rb") as fp:
                        optimization_times = pickle.load(fp)

                    curr_pred_times.append(prediction_times)
                    curr_opti_times.append(optimization_times)


                median_opti_times.append(np.median(curr_opti_times))
                lowperc_opti_times.append(np.percentile(curr_opti_times, q=15))
                uppperc_opti_times.append(np.percentile(curr_opti_times, q=85))

            median_opti_times = np.array(median_opti_times).squeeze()
            lowperc_opti_times = np.array(lowperc_opti_times).squeeze()
            uppperc_opti_times = np.array(uppperc_opti_times).squeeze()
            plt.plot(ns_rffs.squeeze(), median_opti_times, linewidth=2)
            plt.fill_between(ns_rffs.squeeze(), lowperc_opti_times, uppperc_opti_times, alpha=0.3, label='_nolegend_')

        plt.plot(ns_rffs.squeeze(), np.repeat(exact_opti_time_mean, ns_rffs.shape), linewidth=2, linestyle='--')
        plt.fill_between(ns_rffs.squeeze(), exact_lowperc_opti_time, exact_uppperc_opti_time, alpha=0.3, label='_nolegend_')
        plt.ylabel('Opti.\ time [s]')
        plt.xlabel('Number of RFFs')
        plt.legend(['RF-HGP', 'SVGP-HGP', 'Exact HGP'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand',
                   borderaxespad=0, ncol=3, handlelength=1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/new_{experiment}_optimization_times_train_rff_GP_{train_rff_GP}.png", dpi=300, bbox_inches='tight',
                    pad_inches=0)

        plt.show()
experiments_to_be_rerun = []
color_cycle = ['C2', 'C9', 'C3']
style_cycle = ['-', '--', '-.']
for exp_indx, experiment in enumerate(experiments):
    plt.figure(figsize=[8, 3])
    # Load exact GP
    gp_type = 'exact'
    seed = 0
    n_rffs = 0
    with open(f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_train_rff_GP_True_n_rffs_{n_rffs}.txt",
              "rb") as fp:
        reference_signals = pickle.load(fp)

    exact_mu = np.array(reference_signals)

    for type_indx, gp_type in enumerate(gp_types):

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

        plt.plot(ns_rffs.squeeze(), median_mu_errors, linewidth=2, color=color_cycle[type_indx], linestyle=style_cycle[type_indx])
        plt.fill_between(ns_rffs.squeeze(), lowperc_mu_errors, uppperc_mu_errors, alpha=0.3, label='_nolegend_', color=color_cycle[type_indx])
    plt.ylabel(r'$\mathrm{RMSE}_{\%}$ mean')
    plt.xlabel('$m$')
    plt.yscale('log')
    plt.grid(visible=True, which='both')
    plt.legend(["RFF-HGP", "SVGP-HGP"], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand',
                   borderaxespad=0, ncol=3, handlelength=1.0)
    plt.tight_layout()
    plt.savefig(f"figures/new_{experiment}_all_mu_errors_train_rff_GP_{train_rff_GP}.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

print(experiments_to_be_rerun)
for exp_indx, experiment in enumerate(experiments):
    print(experiment)
    # Load exact GP
    gp_type = 'exact'
    seed = 0
    n_rffs = 0
    with open(f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_train_rff_GP_True_n_rffs_{n_rffs}.txt",
              "rb") as fp:
        uncertainties = pickle.load(fp)

    exact_var = np.array(uncertainties)
    plt.figure(figsize=[8, 3])


    for type_indx, gp_type in enumerate(gp_types):
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
        plt.plot(ns_rffs.squeeze(), median_var_errors, linewidth=2, color=color_cycle[type_indx], linestyle=style_cycle[type_indx])
        plt.fill_between(ns_rffs.squeeze(), lowperc_var_errors, uppperc_var_errors, alpha=0.3, label='_nolegend_', color=color_cycle[type_indx])
    plt.ylabel(r'$\mathrm{RMSE}_{\%}$ var.')
    plt.xlabel('$m$')
    plt.yscale('log')
    plt.legend(["RFF-HGP", "SVGP-HGP"], bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), loc='lower left',
               mode='expand',
               borderaxespad=0, ncol=3, handlelength=1.0)

    plt.grid(visible=True, which='both')
    plt.tight_layout()
    plt.savefig(f"figures/new_{experiment}_all_var_errors_train_rff_GP_{train_rff_GP}.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()
