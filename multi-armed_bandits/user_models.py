import copy
import logging
import time
import pyro  # Use time to add delay in the interactive scenario
import torch
import pyro.distributions as dist

import logistic_regression_pyro as lrp
import thompson_sampling_probabilities as tsp
from acquisition_functions import (
    get_arm_with_thompson_sampling,
    get_arm_with_largest_mean,
)


# User models are responsible for generating (simulated) user actions.
def no_lookahead(task_params, reward, arm, data, t, inits, advanced_data = None,data_gen = None,adv_lr_fit = None,debug=False):
    x_arms = task_params["x_arms"]
    z_ = x_arms.new_zeros(x_arms.size()[0])
    return_dict = {"direct_feedback": 1, "a0": -1, "a1": -1, "p_a0": z_, "p_a1": z_}

    return reward, return_dict


def lookahead_irl_ts_avearms_naive(
    task_params, reward, arm, data, t, inits, advanced_data = None,data_gen = None,debug=False,adv_lr_fit = None
):
    # NOTE: currently this modifies the data-dict argument!
    #       It's fine if data-dict is not used later without recreating it.

    beta = task_params["beta"]
    x_arms = task_params["x_arms"]
    reward_probs = task_params["reward_probs"]
    # case y = 0
    data["x"] = torch.cat((data["x"], x_arms[arm : (arm + 1),]), dim=0)
    data["y"] = torch.cat((data["y"], data["y"].new_zeros(1)), dim=0)

    lr_fit = lrp.fit_logistic_regression(data, inits)
    a0, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a0 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    # case y = 1
    data["y"][-1] = data["y"].new_ones(())

    lr_fit = lrp.fit_logistic_regression(data, inits)
    a1, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a1 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    r_a0 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a0
    r_a1 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a1
    a1_vs_a0_ = beta * (r_a0 - r_a1)
    prob_a1_vs_a0_ = 1.0 / (1.0 + torch.exp(a1_vs_a0_))

    action = torch.bernoulli(prob_a1_vs_a0_)
    return_dict = {"direct_feedback": 0, "a0": a0, "a1": a1, "p_a0": p_a0, "p_a1": p_a1}

    if debug is True:
        return_dict["prob_a1_vs_a0"] = prob_a1_vs_a0_
        return_dict["r_a0"] = r_a0
        return_dict["r_a1"] = r_a1

    return action, return_dict


def no_lookahead_irl_ts_avearms_naive(
    task_params, reward, arm, data, t, inits, advanced_data = None,adv_lr_fit = None,debug=False,data_gen = None
):
    # NOTE: currently this modifies the data-dict argument!
    #       It's fine if data-dict is not used later without recreating it.

    beta = task_params["beta"]
    x_arms = task_params["x_arms"]
    reward_probs = task_params["reward_probs"]
    # case y = 0
    data["x"] = torch.cat((data["x"], x_arms[arm : (arm + 1),]), dim=0)
    data["y"] = torch.cat((data["y"], data["y"].new_zeros(1)), dim=0)

    lr_fit = lrp.fit_logistic_regression(data, inits)
    a0, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a0 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    # case y = 1
    data["y"][-1] = data["y"].new_ones(())

    lr_fit = lrp.fit_logistic_regression(data, inits)
    a1, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a1 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    r_a0 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a0
    r_a1 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a1
    a1_vs_a0_ = beta * (r_a0 - r_a1)
    prob_a1_vs_a0_ = 1.0 / (1.0 + torch.exp(a1_vs_a0_))

    action = torch.bernoulli(prob_a1_vs_a0_)
    return_dict = {"direct_feedback": 0, "a0": a0, "a1": a1, "p_a0": p_a0, "p_a1": p_a1}

    if debug is True:
        return_dict["prob_a1_vs_a0"] = prob_a1_vs_a0_
        return_dict["r_a0"] = r_a0
        return_dict["r_a1"] = r_a1

    # ! returns reward, so all the above is dummy code ^
    return reward, return_dict

def theta_teach_vanilla( task_params, reward, arm, data, t, inits, advanced_data = None,adv_lr_fit = None, data_gen = None,debug=False):
    beta = task_params["beta"]
    x_arms = task_params["x_arms"]
    reward_probs = task_params["reward_probs"]
    # case y = 0
    data["x"] = torch.cat((data["x"], x_arms[arm : (arm + 1),]), dim=0)
    data["y"] = torch.cat((data["y"], data["y"].new_zeros(1)), dim=0)

    lr_fit = lrp.fit_logistic_regression(data, inits)

    logpdf_0 = lr_fit["posterior"].log_prob(torch.tensor(task_params["w"]))

    a0, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a0 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    # case y = 1
    data["y"][-1] = data["y"].new_ones(())

    lr_fit = lrp.fit_logistic_regression(data, inits)

    logpdf_1 = lr_fit["posterior"].log_prob(torch.tensor(task_params["w"]))
    
    a1, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a1 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    r_a0 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a0
    r_a1 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a1
    a1_vs_a0_ = beta * (r_a0 - r_a1)
    prob_a1_vs_a0_ = 1.0 / (1.0 + torch.exp(a1_vs_a0_))

    action = torch.bernoulli(prob_a1_vs_a0_)
    return_dict = {"direct_feedback": 0, "a0": a0, "a1": a1, "p_a0": p_a0, "p_a1": p_a1}

    if debug is True:
        return_dict["prob_a1_vs_a0"] = prob_a1_vs_a0_
        return_dict["r_a0"] = r_a0
        return_dict["r_a1"] = r_a1

    return 1 if logpdf_1 > logpdf_0 else 0, return_dict

def theta_teach_advanced(task_params, reward, arm, data, t, inits,advanced_data = None,adv_lr_fit = None, data_gen = None, debug=False):
    # ! change up the data object

    # FIRST, do a regular lookahead run to get p_a0 and p_a1
    # THEn, use that data and knowledge of the student to do smarter stuff, using student logistic model
    #then generate the "next" version of probabilities 

    beta = task_params["beta"]
    x_arms = task_params["x_arms"] # change
    reward_probs = task_params["reward_probs"]

    _, meta = lookahead_irl_ts_avearms_naive(task_params, reward, arm,data , t, inits, advanced_data = None,debug=False)

    # case y = 0: 

    data = data_gen(0, meta)
    lr_fit = lrp.fit_logistic_regression(data, adv_lr_fit)

    a0, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a0 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )
    
    # case y = 1: 

    data = data_gen(1, meta)
    lr_fit = lrp.fit_logistic_regression(data, adv_lr_fit)

    a1, (z_mean, z_chol) = get_arm_with_thompson_sampling(lr_fit, x_arms, key_args={})
    p_a1 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
        z_mean, z_chol, 1000
    )

    r_a0 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a0
    r_a1 = torch.log(reward_probs / (1 - reward_probs)).double() @ p_a1
    a1_vs_a0_ = beta * (r_a0 - r_a1)
    prob_a1_vs_a0_ = 1.0 / (1.0 + torch.exp(a1_vs_a0_))

    action = torch.bernoulli(prob_a1_vs_a0_)
    return_dict = {"direct_feedback": 0, "a0": a0, "a1": a1, "p_a0": p_a0, "p_a1": p_a1}

    if debug is True:
        return_dict["prob_a1_vs_a0"] = prob_a1_vs_a0_
        return_dict["r_a0"] = r_a0
        return_dict["r_a1"] = r_a1

    return action, return_dict

def lookahead_irl_ts_avearms_multi2_naive(
    task_params, reward, arm, data, t, inits, debug=False,adv_lr_fit = None
):
    x_arms = task_params["x_arms"]
    n_steps = task_params["n_steps"]
    w_opt = task_params["w"]
    beta = task_params["beta"]

    # This function will recursively simulate the tree forward to compute the
    # discounted transition probabilities and sum them to get the leaf values.
    # For TS, this uses virtual arms to traverse the tree! Not the true arms.
    def expand_lookahead(x_arm, data, inds, task_params, inits):
        data_0 = copy.deepcopy(data)
        data_1 = copy.deepcopy(data)

        x_arms = task_params["x_arms"]
        gamma = task_params["gamma"]

        if x_arm.dim() == 1:
            x_arm = x_arm.unsqueeze(0)
        data_0["x"] = torch.cat((data_0["x"], x_arm), dim=0)
        data_1["x"] = torch.cat((data_1["x"], x_arm), dim=0)

        data_0["y"] = torch.cat((data_0["y"], data_0["y"].new_zeros(1)), dim=0)
        data_1["y"] = torch.cat((data_1["y"], data_1["y"].new_ones(1)), dim=0)

        # ! LR fit returns estimated thetas , the theacher can just optimize these instead
        lr_fit0 = lrp.fit_logistic_regression(data_0, inits)
        a0, (z_mean, z_chol) = get_arm_with_thompson_sampling(
            lr_fit0, x_arms, key_args={}
        )
        p_a0 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
            z_mean, z_chol, 1000
        )

        ave_arm_a0 = p_a0 @ x_arms

        lr_fit1 = lrp.fit_logistic_regression(data_1, inits)
        a1, (z_mean, z_chol) = get_arm_with_thompson_sampling(
            lr_fit1, x_arms, key_args={}
        )
        p_a1 = tsp.estimate_gaussian_thompson_sampling_probabilities_rbmc(
            z_mean, z_chol, 1000
        )

        ave_arm_a1 = p_a1 @ x_arms

        if len(inds) > 2:
            # not at leaf
            n_branching = len(inds) // 2
            inds_0 = inds[0:n_branching]
            inds_1 = inds[n_branching : len(inds)]
            # note: p_a0 and p_a1 are automatically broadcast as needed
            # note: gamma is applied also to the first branch (for 1-step LA,
            #       if gamma is not 1, this will give different result from the
            #       dedicated 1-step methods)
            return gamma * torch.cat(
                (
                    p_a0
                    + expand_lookahead(
                        ave_arm_a0, data_0, inds_0, task_params, lr_fit0
                    ),
                    p_a1
                    + expand_lookahead(
                        ave_arm_a1, data_1, inds_1, task_params, lr_fit1
                    ),
                ),
                dim=0,
            )
        else:
            # at leaf
            return gamma * torch.stack((p_a0, p_a1))

    inds = list(range(2 ** n_steps))
    P = expand_lookahead(x_arms[arm : (arm + 1),], data, inds, task_params, inits)

    n_branching = len(inds) // 2
    inds_0 = inds[0:n_branching]
    inds_1 = inds[n_branching : len(inds)]

    logits = x_arms @ w_opt
    q_val0 = torch.max(P[inds_0,] @ logits)
    q_val1 = torch.max(P[inds_1,] @ logits)

    a1_vs_a0 = beta * (q_val0 - q_val1)
    prob_a1_vs_a0 = 1.0 / (1.0 + torch.exp(a1_vs_a0))
    action = torch.bernoulli(prob_a1_vs_a0)

    return_dict = {"direct_feedback": 0, "P": P}
    if debug is True:
        return_dict["prob_a1_vs_a0"] = prob_a1_vs_a0
    return action, return_dict
