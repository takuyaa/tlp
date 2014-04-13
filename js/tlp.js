// Copyright (c) 2014 Takuya Asano All Rights Reserved.

/**
 * Two-Layer Perceptron (TLP) for Regression
 * @fileoverview Two-Layer perceptron implementation. This is often called 'Three-Layer' perceptron, for structure of network
 */
(function() {

    "use strict";

    // Hyper parameters
    var D = 2;   // Dimension of x includes bias
    var M = 4;   // Number of hidden units includes bias
    var K = 1;   // Dimension of y

    // Training conditions
    var iterCounts = 0; // number of iteration (for coefficient in learning)
    var eta = function(i) {  // Coefficient in learning
        return 0.1;
    };

    // Weight
    var w_ji = [];  // layer 1 weigth matrix
    var w_kj = [];  // layer 2 weigth matrix


    /**
     * Initialize network
     * @param {Number} hiddenUnits number of hidden units (default: 4)
     * @param {Function} learningCoefficient function returns eta value (default: return 0.1)
     * @param {Function} weightInitFunction function which initialize weights (default: Math.random())
     */
    var init = function(hiddenUnits, learningCoefficient, weightInitFunction) {
        iterCounts = 0;

        if (weightInitFunction == null) {
            weightInitFunction = Math.random;
        }
        if (hiddenUnits != null) {
            M = hiddenUnits;
        }
        if (learningCoefficient != null) {
            eta = learningCoefficient;
        }

        initWji(weightInitFunction);
        initWkj(weightInitFunction);
    };


    /**
     * Create units of input layer
     * @param {Number} x input
     * @returns {Function} function which returns i th unit's output
     */
    var phi = function(x) {

        // 1st element is an output of a bias unit
        var inputUnits = [1, x];

        return function(i) {
            return inputUnits[i];
        };
    };


    /**
     * Calculate an output of j th hidden unit with bias unit
     * @param {Function} x function which returns i th unit's output
     * @param {Number} j index of z_j hidden unit
     * @returns {Number} output of j th hidden unit
     */
    var z_j = function(x, j) {
        if (j == 0) {
            // Bias hidden unit
            return 1;
        }
        var a = 0;
        for (var i = 0; i < D; i++) {
            a += w_ji[j][i] * x(i);
        }
        var z = tanh(a);
        return z;
    };

    var z = function(x) {
        var hiddenOutputs = [];
        for (var j = 0; j < M; j++) {
            hiddenOutputs.push(z_j(x, j));
        }
        return hiddenOutputs;
    };

    /**
     * Calculate an 'feed forward' output of k th output unit
     * @param {Function} x function which returns i th unit's output
     * @param {Number} k index of y_k output unit
     * @param {Array} z precomputing z (optional)
     * @returns {Number} output of k th output unit
     */
    var y_k = function(x, k, z) {
        var ak = 0;
        for (var j = 0; j < M; j++) {
            var zj;
            if (z == null || z[j] == null) {
                // Calculate z_j
                zj = z_j(x, j);
            } else {
                // Use pre-computing z_j
                zj = z[j];
            }
            ak += w_kj[k][j] * zj;
        }
        var y = ak; // Linear output function
        return y;
    };


    var y = function(z) {
        var outputs = [];
        for (var k = 0; k < K; k++) {
            outputs.push(y_k(null, k, z));
        }
        return outputs;
    };


    /**
     * Train network by backpropagation manner for 1 given target
     * @param {Function} x function which returns i th unit's output
     * @param {Number} target target value
     */
    var backPropagation = function(x, target) {

        // Pre-computing z_j
        var z = [];
        for (var j = 0; j < M; j++) {
            if (j == 0) {
                z.push(1);
            } else {
                z.push(z_j(x, j));
            }
        }

        // Pre-computing delta_k (using pre-computing z)
        var delta_k = [];
        for (var k = 0; k < K; k++) {
            var outputError = y_k(x, k, z) - target;
            delta_k.push(outputError);
        }

        // Train hidden -> output weight w_kj
        var w_kj_new = [];
        for (k = 0; k < K; k++) {
            for (j = 0; j < M; j++) {
                if (w_kj_new[k] == null) {
                    w_kj_new[k] = [];
                }
                var diff_kj = eta(iterCounts) * delta_k[k] * z[j];
                w_kj_new[k][j] = w_kj[k][j] - diff_kj;
            }
        }

        // Train input -> hidden weight w_ji
        var w_ji_new = [];
        for (j = 0; j < M; j++) {
            for (var i = 0; i < D; i++) {
                if (w_ji_new[j] == null) {
                    w_ji_new[j] = [];
                }
                var delta_j_part = 0;
                for (k = 0; k < K; k++) {
                    delta_j_part += w_kj[k][j] * delta_k[k];
                }
                var delta_j = delta_j_part * (1 - Math.pow(z[j], 2));
                var diff_ji = eta(iterCounts) * x(i) * delta_j;
                w_ji_new[j][i] = w_ji[j][i] - diff_ji;
            }
        }

        // Update weights
        w_kj = w_kj_new;
        w_ji = w_ji_new;
    };


    /**
     * Calcurate sum of square error function for 1 given target
     * @param {Function} x function which returns i th unit's output
     * @param {Number} target target value
     * @returns {Number} sum of square error
     */
    var error = function(x, target) {
        var sum_err = 0;
        var err = 0;
        for (var k = 0; k < K; k++) {
            err = y_k(x, k) - target;
            sum_err += Math.pow(err, 2);
        }
        return sum_err;
    };


    /**
     * Calcurate summation of error function about each given training datum
     * @param {Array} trainingSet Training data array of object that contains key 'x' and 'y'
     * @returns {Number} sum of error function
     */
    var sumOfError = function(trainingSet) {
        var e_sum = 0;
        for (var n = 0; n < trainingSet.length; n++) {
            var target = trainingSet[n].y;
            var x = phi(trainingSet[n].x);
            e_sum += error(x, target);
        }
        return e_sum;
    };


    /**
     * Initialize weight w_ji by given function
     * @param {Function} f function for initializing
     */
    var initWji = function(f) {
        for (var j = 0; j < M; j++) {
            w_ji[j] = [];
            for (var i = 0; i < D; i++) {
                w_ji[j][i] = f();
            }
        }
    };


    /**
     * Initialize weight w_kj by given function
     * @param {Function} f function for initializing
     */
    var initWkj = function(f) {
        for (var k = 0; k < K; k++) {
            w_kj[k] = [];
            for (var j = 0; j < M; j++) {
                w_kj[k][j] = f();
            }
        }
    };


    /**
     * Train network for 1 given training datum
     */
    var train = function(x_n, y_n) {
        iterCounts++;
        var x = phi(x_n);
        backPropagation(x, y_n);  // Train
    };


    /**
     * Calculate mathematical function tanh(a)
     * @param {Number} a tanh parameter
     * @return {Number} tanh(a) value
     */
    var tanh = function(a) {
        var exp_a = Math.exp(a);
        var exp_m_a = Math.exp(-a);

        var h = (exp_a - exp_m_a) / (exp_a + exp_m_a);

        if (!Number.isFinite(h) && a > 0) {
            return 1;
        } else if (!Number.isFinite(h) && a < 0) {
            return -1;
        } else if (h >= 1) {
            return 1;
        } else if (h <= -1) {
            return -1;
        }

        return h;
    };

    /**
     * Counter number of weight updates
     * @return {Number} number of weight updates
     */
    var trainCount = function() {
        return iterCounts;
    };


    /**
     * Public methods or fields of global object TLP
     */
    var TLP = {
        'phi': phi,
        'z': z,
        'y': y,
        'init': init,
        'train': train,
        'trainCount': trainCount,
        'sumOfError': sumOfError
    };

    if ('undefined' == typeof module) {
	    // In browser
	    window.TLP = TLP;
    } else {
	    // In node
	    module.exports = TLP;
    }

})();
