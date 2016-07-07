/**
 * Created by Attila Bagoly <battila93@gmail.com> on 6/9/16.
 */

(function(factory){
    require.config({
        paths: {
            d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min"
        }
    });

    define(['d3'], function(d3){
        return factory({}, d3);
    });
}(function(NeuralNetwork, d3) {

    // https://github.com/wbkd/d3-extended
    d3.selection.prototype.moveToFront = function() {
        return this.each(function(){
            this.parentNode.appendChild(this);
        });
    };
    d3.selection.prototype.moveToBack = function() {
        return this.each(function() {
            var firstChild = this.parentNode.firstChild;
            if (firstChild) {
                this.parentNode.insertBefore(this, firstChild);
            }
        });
    };

    var style = {
        "neuron": {
            "colors": {
                "input": "#00A000",
                "hidden": "#0000C7",
                "output": "#F6BD00",
                "bias": "#8F686F"
            },
            "mouseon": {
                "change_radius": 2,
                "alpha": 0.2
            }
        },
        "synapse": {
            "colors": {
                "negative": "#00005E",
                "positive": "#5E0000"//"#FF4B00"
            },
            "default_width_range": [0.5, 5],
            "width_range": [0.5, 5],
            "default_alpha": 0.7,
            "alpha": 0.7,
            "mouseon": {
                "width_range": [0.5, 10],
                "alpha": 0.1
            }
        },
        "variables": {
            "labels_layer0_padding": 0.03
        },
        "legend": {
            "pos": {"x": 150, "y": 10},
            "rect": {"width": 10, "height":10},
            "dy": 20,
            "padding": 10
        }
    };

    var canvas;

    var getNeuronNumber = function(net, layer_index){
        return Number(Object.keys(net["layout"]["layer_"+layer_index]).length-1);
    };

    var getNeuronsAttr = function (net, num_layers, layer_index) {
        var numn = getNeuronNumber(net, layer_index);
        var neuronsAttr = Array(numn);
        for(var i=0;i<numn;i++){
            neuronsAttr[i] = {
                "position": {
                    "x": (layer_index + 0.5) * canvas.width / (num_layers),
                    "y": (i + 0.5) * canvas.height / numn
                },
                "radius": canvas.height/(numn+(numn>5?0:5))/4,
                "type":  (i==(numn-1) ? "bias" : (layer_index==0 ? "input" : "hidden")),
                "neuron": i,
                "layer": layer_index,
            };
            if (layer_index==(num_layers-1)){
                neuronsAttr[i]["type"] = "output";
            }
        }
        return neuronsAttr;
    };

    var getWeights = function(net, layer, neuron){
        var neuron = net["layout"]["layer_"+layer]["neuron_"+neuron];
        if (neuron["nsynapses"]!=0) return neuron["weights"];
        return [];
    };

    var getMinMaxWeight = function(net, num_layers){
        var max = -1e30;
        var min =  1e30;
        var tmp;
        for(var i=0;i<num_layers;i++){
            for(var j=0;j<getNeuronNumber(net, i);j++){
                tmp = d3.max(getWeights(net, i, j));
                if (max < tmp) max = tmp;
                tmp = d3.min(getWeights(net, i, j));
                if (min > tmp) min = tmp;
            }
        }
        return {"min": min, "max": max};
    };

    var getSynapses = function(net, layer_index, neuron, pos, layer2){
        var weights  = getWeights(net, layer_index, neuron);
        var synapses = Array(weights.length);
        for(var i in weights){
            synapses[i] = {
                "layer": layer_index,
                "neuron": neuron,
                "nextlayer_neuron": i,
                "pos": [pos, layer2[i].position],
                "weight": weights[i],
                "type":   (weights[i]<0 ? "negative" : "positive")
            };
        }
        return synapses;
    };

    var getInputLabels = function(net, layer0){
        var labels = net["variables"];
        labels.push("Bias node");
        var variables = Array(labels.length);
        for(var i in layer0){
            variables[i] = {
                "x": layer0[i].position.x-style["variables"]["labels_layer0_padding"]*canvas.width,
                "y": layer0[i].position.y,
                "text": labels[i] + ":"
            };
        }
        return variables;
    };

    var drawInputLabels = function(group){
        group.append("text")
            .text(function(d){return d[1].text;})
            .attr("x", function(d){return d[1].x-this.getComputedTextLength();})
            .attr("y", function(d){return d[1].y+0.25*this.getBBox().height;});
    };

    var drawNeurons = function (svg, net, neuronsattr, layer_num, input_variable_labels, deepNet) {
        if (input_variable_labels!==undefined){
            var dat = d3.zip(neuronsattr, getInputLabels(net, neuronsattr));
        } else {
            var dat = d3.zip(neuronsattr, Array(neuronsattr.length));
        }
        var group = svg.append("g").attr("id", "layer_"+layer_num).attr("class", "layer").selectAll("g")
            .data(dat)
            .enter()
            .append("g").attr("id", function(d){return "neuron_"+layer_num+""+d[0].neuron;});
        group.append("circle")
            .attr('r',     function(d){return d[0].radius})
            .attr('cx',    function(d){return d[0].position.x;})
            .attr('cy',    function(d){return d[0].position.y;})
            .style("fill", function(d){return style["neuron"]["colors"][d[0].type];});
        if (input_variable_labels!==undefined){
            drawInputLabels(group)
        }
        if (deepNet) return;
        animate(svg, group);
    };

    var scaleSynapsisPos = d3.scale.linear();
    var scaleSynapsisNeg = d3.scale.linear();

    var synapse = d3.svg.line()
        .x(function(d){return d.x;})
        .y(function(d){return d.y;})
        .interpolate("linear");

    var drawSynapses = function(svg, net, layer1, layer1_index, layer2){
        for(var idx in layer1){
            var synapses = getSynapses(net, layer1_index, idx, layer1[idx].position, layer2);
            svg.select("g#neuron_"+layer1_index+""+idx)
                .selectAll("path")
                .data(synapses)
                .enter()
                .append("path").moveToBack()
                .attr("d", function(d){return synapse(d.pos);})
                .attr("stroke", function(d){return style["synapse"]["colors"][d.type]})
                .attr("stroke-width", function(d){
                    return d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                })
                .attr("stroke-opacity", style["synapse"]["alpha"]);
        }
    };

    var animate = function(svg, group){
        style.synapse.width_range = Object.assign({}, style.synapse.default_width_range);
        style.synapse.alpha = Object.assign({}, style.synapse.default_alpha);
        group.on('mouseover', function(d) {
            scaleSynapsisPos.range(style["synapse"]["mouseon"]["width_range"]);
            scaleSynapsisNeg.range(style["synapse"]["mouseon"]["width_range"]);
            var self = d3.select(this).moveToFront().transition();
            self.selectAll("path")
                .style("stroke-opacity", 1)
                .attr("stroke-width", function(d){
                    return d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                });
            self.selectAll("circle")
                .style("fill-opacity", 1)
                .attr("r", function(d){return d[0].radius*style["neuron"]["mouseon"]["change_radius"]});
            self.selectAll("text")
                .attr("x", function(d){return d[1].x-d[0].radius-this.getComputedTextLength();});

            var allbutnotthis = svg.selectAll("g.layer").selectAll("g")
                .filter(function(x){return !(d[0].neuron==x[0].neuron&&d[0].layer==x[0].layer);}).transition();
            allbutnotthis.selectAll("circle").filter(function(x){return (d[0].layer+1)!=x[0].layer})
                .style("fill-opacity", style["neuron"]["mouseon"]["alpha"])
                .attr("r", function(d){return d[0].radius})
            allbutnotthis.selectAll("path")
                .style("stroke-opacity", style["synapse"]["mouseon"]["alpha"]);
        });
        group.on('mouseout', function(d){
            scaleSynapsisPos.range(style["synapse"]["width_range"]);
            scaleSynapsisNeg.range(style["synapse"]["width_range"]);
            var gg = svg.selectAll("g.layer").selectAll("g").transition();
            gg.selectAll("circle")
                .style("fill-opacity", 1)
                .attr("r", function(d){return d[0].radius;});
            gg.selectAll("path")
                .style("stroke-opacity", style["synapse"]["alpha"])
                .attr("stroke-width", function(d){
                    return d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                });
            gg.selectAll("text")
                .attr("x", function(d){return d[1].x-this.getComputedTextLength();});
        });
    };

    var drawLegend = function(svg){
        var labels = [
            {"c": style["synapse"]["colors"]["positive"], "txt": "Positive weight"},
            {"c": style["synapse"]["colors"]["negative"], "txt": "Negative weight"}
        ];
        var attr = style["legend"];

        var container = svg.append("g").attr("id", "legend");
        container.selectAll("g")
            .data(labels)
            .enter()
            .append("g")
            .each(function(d, i){
                var g = d3.select(this);
                g.append("rect")
                    .attr("x", canvas.width-attr.pos.x)
                    .attr("y", attr.pos.y+i*attr.dy)
                    .attr("width", attr.rect.width)
                    .attr("height", attr.rect.height)
                    .style("fill", function(d){return d.c;});
                g.append("text")
                    .attr("x", canvas.width-attr.pos.x+attr.rect.width+attr.padding)
                    .attr("y", attr.pos.y+i*attr.dy+attr.rect.height)
                    .text(function(d){return d.txt;})
                    .style("fill", function(d){return d.c;});
            });
    };

    NeuralNetwork.draw = function (divid, netobj) {
        var svg, net;

        var div = d3.select("#"+divid);
        canvas = {
            width:  div.property("style")["width"],
            height: div.property("style")["height"]
        };

        var deepNet;
        if ("layers" in netobj && "synapses" in netobj){
            net = transformDeepNetObject(netobj);
            style.synapse.width_range = [5/netobj["synapses"]["synapses"].length, 50/netobj["synapses"]["synapses"].length];
            style.synapse.alpha = 0.9;
            scaleSynapsisPos.range(style["synapse"]["width_range"]);
            scaleSynapsisNeg.range(style["synapse"]["width_range"]);
            deepNet = true;
        } else {
            net = netobj;
            style.synapse.width_range = Object.assign({}, style.synapse.default_width_range);
            style.synapse.alpha = Object.assign({}, style.synapse.default_alpha);
            scaleSynapsisPos.range(style["synapse"]["width_range"]);
            scaleSynapsisNeg.range(style["synapse"]["width_range"]);
            deepNet = false;
        }

        /*net = netobj;
        style.synapse.width_range = style.synapse.default_width_range;
        style.synapse.alpha = style.synapse.default_alpha;
        scaleSynapsisPos.range(style["synapse"]["width_range"]);
        scaleSynapsisNeg.range(style["synapse"]["width_range"]);*/

        svg = div.append("svg")
            .attr("id", "svg_"+divid)
            .attr("width", canvas.width)
            .attr("height", canvas.height);
        Object.keys(canvas).forEach(function (key) {
            canvas[key] = Number(canvas[key].replace("px",""))
        });

        var num_layers = Number(net["layout"]["nlayers"]);

        scaleSynapsisPos.domain([0,getMinMaxWeight(net, num_layers).max]);
        scaleSynapsisNeg.domain([0, Math.abs(getMinMaxWeight(net, num_layers).min)]);
        var zoom = d3.behavior.zoom()
            .scaleExtent([1, 20])
            .on("zoom", function(){
                svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
            });
        svg = svg
            .on("dblclick", function(){
                zoom.scale(1);
                zoom.translate([0, 0]);
                svg.transition().attr("transform", "translate(0,0)scale(1)");
            })
            .append("g").call(zoom).append("g");

        var layers = Array(num_layers);
        for(var i=0;i<num_layers;i++){
            layers[i] = getNeuronsAttr(net, num_layers, i);

        }
        for(i=0;i<num_layers;i++) {
            drawNeurons(svg, net, layers[i], i, i==0 ? true : undefined, deepNet);
            drawSynapses(svg, net, layers[i], i, layers[i + 1]);
        }
        drawLegend(svg);
    };


    var transformDeepNetObject = function(deepnet){
        vars = deepnet["variables"];
        vars.push("Bias node");
        var layers = deepnet["layers"];
        var synapses = deepnet["synapses"]["synapses"];
        var layout = {
            layer_0: {
                nneurons: vars.length
            }
        };
        var nodes;
        for(var j=0;j<vars.length;j++){
            nodes = Number(layers[0]["Nodes"]);
            layout["layer_0"]["neuron_"+j]={
                nsynapses: nodes,
                weights: synapses.slice(j*nodes, (j+1)*nodes)
            }
        }
        for(var i=0; i<(layers.length-1);i++){
            layout["layer_"+(i+1)] = {
                nneurons: Number(layers[i]["Nodes"])
            };
            nodes = Number(layers[i+1]["Nodes"]);
            for(var j=0; j<Number(layers[i]["Nodes"]);j++){
                layout["layer_"+(i+1)]["neuron_"+j] = {
                    nsynapses: nodes,
                    weights: synapses.slice(j*nodes, (j+1)*nodes)
                }
            }
        }
        layout["layer_"+(i+1)] = {
            nneurons: Number(layers[i]["Nodes"])
        };
        for(var j=0; j<Number(layers[i]["Nodes"]);j++){
            layout["layer_"+(i+1)]["neuron_"+j] = {
                nsynapses: 0
            }
        }
        layout["nlayers"] = i+2;
        var net = {
            variables: vars,
            layout: layout
        };
        return net;
    };

    var drawDeepNetNeurons = function (context, neuronsattr) {
        for(var i=0;i<neuronsattr.length;i++){
            context.beginPath();
            context.arc(neuronsattr[i].position.x, neuronsattr[i].position.y, neuronsattr[i].radius, 0, 2*Math.PI);
            context.fillStyle = style["neuron"]["colors"][neuronsattr[i].type];;
            context.fill();
            context.closePath();
        }
    };

    var drawDeepNetSynapses = function(ctx, net, layer1, layer1_index, layer2){
        var idx, si, d;
        for(idx in layer1) {
            var synapses = getSynapses(net, layer1_index, idx, layer1[idx].position, layer2);
            for(si in synapses){
                d = synapses[si];
                ctx.beginPath();
                ctx.moveTo(d.pos[0].x, d.pos[0].y);
                ctx.lineTo(d.pos[1].x, d.pos[1].y);
                ctx.lineWidth = d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                ctx.strokeStyle = style["synapse"]["colors"][d.type];
                ctx.stroke();
                ctx.closePath();
            }
        }
    };

    NeuralNetwork.drawDeepNetwork = function (divid, netobj) {
        var div = d3.select("#"+divid);
        canvas = {
            width:  div.property("style")["width"],
            height: div.property("style")["height"]
        };

        /** ADDING EXTRA LAYERS FOR TEST **/
        /** TODO remove this **/
        var N =50;
        netobj["layers"].splice(1, 0, {"Nodes" : N});
        for(var i=0;i<100*N;i++) {
            netobj["synapses"]["synapses"].push((Math.random() > 0.5 ? -1 : 1) * Math.random() * 10);
        }
        for(var k=0;k<10;k++){
            netobj["layers"].splice(1, 0, {"Nodes" : N});
            for(var i=0;i<N*N;i++) {
                netobj["synapses"]["synapses"].push((Math.random() > 0.5 ? -1 : 1) * Math.random() * 10);
            }
        }
        for(var i=0;i<N*50;i++) {
            netobj["synapses"]["synapses"].push((Math.random() > 0.5 ? -1 : 1) * Math.random() * 10);
        }
        net = transformDeepNetObject(netobj);

        style.synapse.width_range = [50/netobj["synapses"]["synapses"].length, 1000/netobj["synapses"]["synapses"].length];
        style.synapse.alpha = 0.9;
        scaleSynapsisPos.range(style["synapse"]["width_range"]);
        scaleSynapsisNeg.range(style["synapse"]["width_range"]);

        var context = div.append("canvas")
            .attr("width", canvas.width)
            .attr("height", canvas.height)
            .node().getContext("2d");

        Object.keys(canvas).forEach(function (key) {
            canvas[key] = Number(canvas[key].replace("px",""))
        });

        var num_layers = Number(net["layout"]["nlayers"]);

        var layers = Array(num_layers);
        for(var i=0;i<num_layers;i++){
            layers[i] = getNeuronsAttr(net, num_layers, i);
        }

        for(i=0;i<num_layers;i++) {
            drawDeepNetSynapses(context, net, layers[i], i, layers[i + 1]);
            drawDeepNetNeurons(context, layers[i]);
        }
    };

    Object.seal(NeuralNetwork);
    return NeuralNetwork;
}));