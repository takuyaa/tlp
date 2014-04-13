// Copyright (c) 2014 Takuya Asano All Rights Reserved.

// Two-Layer Perceptron

// Data
var t = [];  // Training Data
var e = [];  // Error sum of square (for display)

var init = function() {
    t = [];
    e = [];
};

var sample = function(min, max) {
    return Math.random() * (max - min) + min;
};

var series = function(min, max, N, f) {
    for(var i = 0; i < N; i++) {
        var xi = (i * (max-min) / (N-1)) + min;
        f(xi);
    }
};

var calcError = function(trainingSet, copy) {
    if (copy == null) {
        copy = 1;
    }
    var count = TLP.trainCount();
    var soe = TLP.sumOfError(trainingSet) * copy;
    var averagedError = soe / count;
    return {
        x: count, y: averagedError
    };
};


/**
 * Generate training data by given steps
 * @param {Function} generator function which generates training data in same distance
 * @param {Number} numGenerates number of training data to generate
 * @param {Number} min data range from
 * @param {Number} max data range to
 */
var generateSequentialData = function(generator, numGenerates, min, max) {
    // Prepare training data
    var trainingSet = [];
    series(min, max, numGenerates, function(xi) {
        trainingSet.push({
            x: xi,
            y: generator(xi)
        });
    });
    return trainingSet;
};


/**
 * SGD (Stochastic Gradient Descent) learning
 * @param {Array} trainingSet dataset to train perceptron
 * @param {Number} iteration number of trainings continue
 */
var trainSeq = function(trainingSet, iteration, calcErrorPeriod) {

    trainingSet.map(function(datum) {
        t.push(datum);
    });

    for (var i = 0; i < iteration; i++) {
        for (var n = 0; n < trainingSet.length; n++) {

            // Train
            var xi = trainingSet[n].x;
            var target = trainingSet[n].y;
            TLP.train(xi, target);

            // Calculate Error
            if (TLP.trainCount() % calcErrorPeriod == 0) {
                var error = calcError(t, i+1);
                e.push(error);
            }
        }
    }
};


/**
 * SGD (Stochastic Gradient Descent) learning
 * Training data are sampled between 'min' to 'max'
 * @param {Function} generator function which generates training data
 * @param {Number} N number of sampling
 * @param {Number} min data range from
 * @param {Number} max data range to
 */
var trainRandom = function(generator, N, min, max, calcErrorPeriod) {

    for(var ii = 0; ii < N; ii++) {

        // sampling
        var xi = sample(min, max);

        t.push({
            x: xi,
            y: generator(xi)
        });

        TLP.train(xi, generator(xi));

        // Calcurate Error
        if (TLP.trainCount() % calcErrorPeriod == 0) {
            e.push(calcError(t));
        }
    }
};


/**
 * SGD (Stochastic Gradient Descent) learning
 * You can give arbitrary training datum
 * @param {Number} xi training datum
 * @param {Number} target training datum
 */
var trainOne = function(xi, target) {

    t.push({
        x: xi,
        y: target
    });

    // Train
    TLP.train(xi, target);
};


/**
 * Display solution of regression in browser
 */
var disp = function(step) {

    if (step == null) {
        step = 50;
    }

    var y = [];  // Output of output layer (for display)
    var z = [];  // Output of hideen layer (for display)


    // Sort Training Data

    t.sort(function(t1, t2) {
        if (t1.x < t2.x) {
            return -1;
        } else if (t1.x > t2.x){
            return 1;
        }
        return 0;
    });


    // Output of Trained NN

    var min = t[0].x;
    var max = t[t.length-1].x;
    series(min, max, step, function(xn) {

        // Feed forward
        var x_ = TLP.phi(xn);
        var z_ = TLP.z(x_);
        var y_ = TLP.y(z_);

        y.push({
            x: xn, y: y_[0]
        });

        for (var j = 0; j < z_.length; j++) {
            if (z[j] == null) {
                z[j] = [];
            }
            z[j].push({
                x: xn, y: z_[j]
            });
        }
    });


    // Display Graph

    nv.addGraph(function() {
        var chart = nv.models.lineChart()
                    .useInteractiveGuideline(true);

        chart.xAxis
        .axisLabel("x")
        .tickFormat(d3.format(".03f"));

        chart.yAxis
        .axisLabel("y")
        .tickFormat(d3.format(".03f"));

        var chartData = [
            {
                key: "Training Data",
                values: t,
                color: "#7777ff"
            },
            {
                key: "NN Output",
                values: y,
                color: "#ff7f0e"
            }
        ];

        if (showHiddenOutput) {
            for (var j = 0; j < z.length; j++) {
                chartData.push(
                    {
                        key: 'z_' + j,
                        values: z[j],
                        color: "#2ca02c"
                    }
                );
            }
        }

        d3.select("#input-output")
        .datum(chartData)
        .transition().duration(500).call(chart)
        ;

        nv.utils.windowResize(
            function() {
                chart.update();
            }
        );

        return chart;
    });

    nv.addGraph(function() {
        var chart = nv.models.lineChart()
                    .useInteractiveGuideline(true);

        chart.xAxis
        .axisLabel("iteration")
        .tickFormat(d3.format("d"));

        chart.yAxis
        .axisLabel("error")
        .tickFormat(d3.format('.03f'));

        d3.select("#error")
        .datum([
            {
                key: "Error",
                values: e,
                color: "#7777ff"
            }
        ])
        .transition().duration(500).call(chart);

        nv.utils.windowResize(
            function() {
                chart.update();
            }
        );

        return chart;
    });
};



// Display conditions
var showHiddenOutput = false;  // Display hidden unit output or not
var isRandomSampling = false;  // Lean from random sampling data

var targetFunction;
var x_2 = 'x * x';
var sin_pi_x = 'Math.sin( Math.PI * x )';
var abs_x = 'Math.abs(x)';
var heaviside_x = 'heaviside(x)';

var heaviside = function(x) {
    if (x < 0) {
        return 0
    } else {
        return 1
    }
};

// Write default conditions to forms
$('#target-function').val(sin_pi_x);  // default: sin(pi*x)
$('#M').val(4);
$('#eta').val('0.1');
$('#seq-number-data').val(50);
$('#seq-iteration').val(1000);
$('#random-number-sampling').val(5000);

// 'Generating function' radio button binding
$('.target-radio')
.on('click', function(e) {
    switch (e.currentTarget.id) {
        case 'target-sin':
        $('#target-function').val(sin_pi_x);
        break;
        case 'target-x2':
        $('#target-function').val(x_2);
        break;
        case 'target-abs':
        $('#target-function').val(abs_x);
        break;
        case 'target-heaviside':
        $('#target-function').val(heaviside_x);
        break;
    }
});

// 'Sampling' button binding
$('.sub-nav dd').on('click', function(e) {
    $('.sub-nav dd').removeClass('active');
    $(this).addClass('active');
    if(e.currentTarget.id == 'is-random-sampling') {
        $('#seq-param').css('display', 'none');
        $('#random-param').css('display', '');
        isRandomSampling = true;
    } else {
        $('#seq-param').css('display', '');
        $('#random-param').css('display', 'none');
        isRandomSampling = false;
    };
});


var calculate = function() {
    var targetFuncInput = $('#target-function').val();
    var hiddenUnits = $('#M').val();
    var eta_ = $('#eta').val();
    var set = $('#seq-number-data').val();
    var iteration = ($('#seq-iteration').val());

    init();

    var target;
    try {
        eval('target = function(x) {'
            + 'return ' + targetFuncInput + ';};');
        target();
    } catch (e) {
        alert('Target function input is invalid');
        return;
    }
    targetFunction = target;

    // Initialize Perceptron
    TLP.init(hiddenUnits, function(i) {
        // eta_ = 1 / i;  // Pegasos
        // eta_ = 100 / (1000 + i) + 0.1;
        return parseFloat(eta_);
    });

    if (isRandomSampling) {
        // Train
        var sample = $('#random-number-sampling').val();
        trainRandom(targetFunction, sample, -1, 1, 128);
        set = 128;
    } else {
        // Train
        // Training data are generated in same distance ('set' steps between -1 to 1)
        var trainingSet = generateSequentialData(targetFunction, set, -1, 1);
        trainSeq(trainingSet, iteration, set);
    }

    // Display results
    disp(set);
};

// 'Calculate' button binding
$('#calculate-button').on('click', calculate);

// Calculate regression when this js loaded
calculate();
