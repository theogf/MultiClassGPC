using ArgParse

function parse_commandline()
    s = ArgParseSettings(exc_handler=ArgParse.debug_handler)
    @add_arg_table s begin
        "dataset"
            help = "Dataset to train on"
        "-M", "--indpoints"
            help = "Number of inducing points"
            arg_type = Int
            default = 0
        "-S", "--stochastic"
            help = "Run stochastic updates"
            action = :store_true
        "-b", "--batchsize"
            help = "Size of the minibatches"
            arg_type = Int
            default = 100
        "--autotuning", "-A"
            help = "Autotuning activated or not"
            action = :store_true
        "--point-optimization", "-P"
            help = "Optimize inducing point location"
            action = :store_true
        "--maxiter", "-I"
            help = "Maximum number of iterations (x10 for SVGP)"
            arg_type = Int
            default = 100
        "--XGP"
            help = "Run XGPC"
            action = :store_true
        "--SVGP"
            help = "Run SVGPC"
            action = :store_true
        "--EPGP"
            help = "Run EPGPC"
            action = :store_true
        "--TTGP"
            help = "Run Tensor Train decomposition"
            action = :store_true
        "--AR"
            help = "Run Augment and Reduce"
            action = :store_true
        "--last-state"
            help = "Save the last state of the model"
            action = :store_true
        "--no-writing"
            help = "Do not write the convergence results into files"
            action = :store_true
        "--plot"
            help = "Plot convergence results"
            action = :store_true
        "--nFold"
            help = "Divide the data set in n Folds"
            arg_type = Int
            default = 10
        "--iFold"
            help = "Number of fold which must be estimated (must be less or equal to nFold)"
            arg_type = Int
            default = 10
        "--shared_prior"
            help = "Determines if the GPs share the same prior"
            action = :store_true
    end

    return parse_args(ARGS,s)
end

args = parse_commandline();
