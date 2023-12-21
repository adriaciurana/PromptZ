var DEFAULT_PARAMS = {
    "runtime_config": {
        "max_population": 5,
        "topk_population": 10,
        "iterations": 2,
        "generator_samples": 5,
    },
    "config_name": "",
    "initial_prompt": "",
    "target": "",
};
var WS;
var NODES = [];
var NODES_DICT = {};
var NODE_ID_TO_JSID = {};
var NODE_JSID_TO_ID = {};
var EDGES = [];
var CHROMOSOMES = {};

var NUM_TOTAL_ITERATIONS = 10;

function start_progress_bar() {
    // yellow
    $(".progressBar").addClass("w-[" + 0 + "%] bg-yellow-300");

    // Logic to show the progress bar
    update_progress_bar(0)
}

function update_progress_bar(current_iteration){
    let percentage = current_iteration / NUM_TOTAL_ITERATIONS * 100;

    $(".progressBar").addClass("w-[" + percentage + "%] bg-yellow-300");
}

function finish_progress_bar(current_iteration) {
    let percentage = current_iteration / NUM_TOTAL_ITERATIONS * 100;

    // add class as green color
    $(".progressBar").addClass("w-[" + percentage + "%] bg-green-500");
}

function recompute_topk(){
    let html_topk = "<div>The following prompts are best suited for your purpose -</div>";

    let chromosomes = [];
    for(let chromosome_id in CHROMOSOMES){
        let c = CHROMOSOMES[chromosome_id];
        if(c != null){
            chromosomes.push(c);
        }
        
    }

    chromosomes.sort((c_a, c_b) => {
        return c_b['score'] - c_a['score'];
    });
    
    for(let i = 0; i < DEFAULT_PARAMS["runtime_config"]["topk_population"]; i++){
        let chromosome = chromosomes[i];

        html_topk += "<div class='border w-full text-sm bg-white rounded-md border-[#E0E0E0] pt-5'> <div class='px-5'> <div class='opacity-50 text-black text-sm font-semibold'>Prompt</div> <div class='text-black text-sm font-bold text-wrap'>" + chromosome['prompt'] + "</div> <div class='w-full h-[0px] my-3 opacity-20 border border-black'></div> <div class='flex w-full items-center justify-between'></div> <div class='opacity-50 text-black text-sm font-semibold'>Response</div>  <div class='text-black text-sm font-regular mt-4'>" + chromosome['output'] + "</div> </div> <div class='w-full flex items-center px-5 py-2 mt-4 bg-[#FEFFF5] rounded-bl-[5px] rounded-br-[5px] border border-neutral-200 text-black'> Score: " + chromosome['score'] + " </div> </div>";
    }

    $(".topk-menu .list").html(html_topk);
}


function on_show_node(id){
    let chromosome = CHROMOSOMES[NODE_JSID_TO_ID[id]];
    let chromosome_html = "<div class='p-1'><div class='w-full text-sm bg-zinc-800 rounded-md pt-5'> <div class='px-5'> <div class='opacity-50 text-[#FFFFFF] text-sm font-semibold'>Prompt</div> <div class='text-[#FFFFFF] text-sm font-bold text-wrap'>" + chromosome['prompt'] + "</div> <div class='w-full h-[0px] my-3 opacity-20 border border-[272727]'></div> <div class='flex w-full items-center justify-between'> </div><div class='opacity-50 text-[#FFFFFF] text-sm font-semibold'>Response</div> <div class='text-[#FFFFFF] text-sm font-regular mt-4'>" + chromosome['output'] + "</div> </div> <div class='w-full flex items-center px-5 py-2 mt-4 bg-zinc-900 rounded-bl-[5px] rounded-br-[5px] text-[#FFFFFF]'> Score: " + chromosome['score'] + " </div> </div></div>";
    $(".show-menu").html(chromosome_html);
}


function on_start(){
    let params = DEFAULT_PARAMS; // TODO: copy in the future.

    let target = $("#target").val();
    $(".target-txt").html(target);
    let initial_prompt = $("#initial_prompt").val();
    $(".start-menu").hide();
    $(".start-btn").hide(); // for now
    $(".topk-menu").show();
    $(".hide_btn").removeClass("hidden")        
    $(".d3-component").removeClass("hidden")
    $(".right-side").removeClass("hidden")
    $(".left-side").addClass("overflow-y-scroll")
    $(".left-side").removeClass("bg-white m-16 border border-[#E0E0E0] h-fit")
    $(".d3-component").addClass("w-full bg-black")
    params['initial_prompt'] = initial_prompt;
    params['target'] = target;
    NUM_TOTAL_ITERATIONS = params['runtime_config']['iterations'];
    params['config_name'] = $('#configurations').find(":selected").val();

    send_cmd("run", params); 
}

function on_change_configuration(){
    let option = $('#configurations').find(":selected").val();
    send_cmd("get_default_inputs", {'config_name': option});
}

/* GRAPH FUNCTIONS */
function send_cmd(cmd, params){
    if(typeof(params) == 'undefined'){
        params = {};
    }
    WS.send(JSON.stringify({'cmd': cmd, 'params': params}));
    console.log("send_cmd sent");
}

function init_graph(msg_json){
    // Node main
    let node_main = {
        id: NODES.length, 
        level: 0,
        status: "root", 
        score: "root",
    };
    NODES.push(node_main);
    NODE_ID_TO_JSID[1] = 0;
    NODE_JSID_TO_ID[0] = 1;
    NODES_DICT[1] = node_main;
    CHROMOSOMES[1] = null;

    // Population
    let population = msg_json["population"];
    for(let i = 0; i < population.length; i++){
        let chromosome = population[i];
        let node = {
            id: NODES.length, 
            level: 1,
            status: "ok", 
            score: chromosome.score,
        };
        let current_jsid = NODES.length;
        NODE_ID_TO_JSID[chromosome.id] = current_jsid;
        NODE_JSID_TO_ID[current_jsid] = chromosome.id;
        NODES.push(node);
        NODES_DICT[chromosome.id] = node;
        CHROMOSOMES[chromosome.id] = chromosome;

        // Parse parents
        EDGES.push({
            source: parseInt(NODE_ID_TO_JSID[1]),
            target: current_jsid,
        });
    }
    draw_graph();
    recompute_topk();
    start_progress_bar();
}

function generated_graph(msg_json){
    let iteration = msg_json["iteration"];
    let variations = msg_json["variations"];
    for(let i = 0; i < variations.length; i++){
        let chromosome = variations[i];
        let node = {
            id: NODES.length, 
            level: iteration + 2,
            status: "alive", 
            score: chromosome.score,
        };
        let current_jsid = NODES.length;
        NODE_ID_TO_JSID[chromosome.id] = current_jsid;
        NODE_JSID_TO_ID[current_jsid] = chromosome.id;
        NODES.push(node);
        NODES_DICT[chromosome.id] = node;
        CHROMOSOMES[chromosome.id] = chromosome;

        // Parse parents
        let parents_id;
        if(!Array.isArray(chromosome.parent_id)){
            parents_id = [chromosome.parent_id];
        
        }else{
            parents_id = chromosome.parent_id;
        }



        // Create edges
        for(let j = 0; j < parents_id.length; j++){
            let parent_id = parents_id[j];
            EDGES.push({
                source: parseInt(NODE_ID_TO_JSID[parent_id]),
                target: current_jsid,
            });
        }
    }
    draw_graph();
    recompute_topk();
    update_progress_bar(iteration);

}

function filtered_graph(msg_json){
    let iteration = msg_json["iteration"];
    let current_status_population = msg_json["current_status_population"];
    for(let chromosome_id in current_status_population){
        let value = current_status_population[chromosome_id];
        NODES[NODE_ID_TO_JSID[chromosome_id]].status = (value) ? 'alive' : 'removed';
    }
    draw_graph();
    recompute_topk();
    update_progress_bar(iteration);
}

function results_graph(msg_json){
    // let iteration = msg_json["iteration"];
    let population_ids = msg_json["population_ids"];
    for(let i = 0; i < population_ids.length; i++){
        let chromosome_id = population_ids[i];
        NODES[NODE_ID_TO_JSID[chromosome_id]].status = 'topk';
    }
    draw_graph();
    recompute_topk();
    finish_progress_bar(NUM_TOTAL_ITERATIONS);
}

function add_configurations(msg_json){
    let html = "";
    let names = msg_json['names'];
    for(let i = 0; i < names.length; i++){
        if(i == 0){
            html += '<option selected value="' + names[i] + '">' + names[i] + '</option>';
        }else{
            html += '<option value="' + names[i] + '">' + names[i] + '</option>';
        }
    }
    $("#configurations").html(html);

    on_change_configuration();
}

function get_inputs(msg_json){
    $("#target").val(msg_json['inputs']['target']);
    $("#initial_prompt").val(msg_json['inputs']['initial_prompt']);
}

/* WEB SOCKET */

// Based on https://github.com/cankav/simple_websocket_example

$(document).ready(function () {
    // let params = DEFAULT_PARAMS; // TODO: copy in the future.
    WS = open_ws();
}); // ready end

function open_ws(){
    // websocket on same server with address /websocket
    let ws = new WebSocket("ws://localhost:4003/websocket");
    ws.onopen = function(){
        // Web Socket is connected, send data using send()
        console.log("ws open");
        send_cmd("get_configurations");
        $("#root").removeClass("hide-overflow");
        $("#message-connecting").hide();
    };

    ws.onmessage = function (evt){
        let msg = evt.data;
        let msg_json = JSON.parse(msg);
        let cmd = msg_json["operation"];
        console.log(msg_json);

        switch(cmd){
            // Init the graph
            case "init":
                init_graph(msg_json);
                break;
                
            case "generated":
                generated_graph(msg_json);
                break;

            case "filtered":
                filtered_graph(msg_json);
                break;

            case "results":
                results_graph(msg_json);
                break;

            case "configutations": 
                add_configurations(msg_json);
                break;

            case "inputs": 
                get_inputs(msg_json);
                break;
        }
    };

    ws.onclose = function(){ 
        // websocket is closed, re-open
        console.log("Connection is closed...");
    };

    return ws;
}

/* DRAW GRAPH */
const SIZE = 1200;
const DEFAULT_RADIUS = 20;
let MARGIN = 80;
let GAP = 70;

const D3NODE = document.querySelector("#d3_node");

// Define cola
const D3ADAPTOR = cola.d3adaptor;
const right_side = $(".right-side");
// right_side.removeClass("hidden")
let width_size = right_side.width();
let height_size = right_side.height();

const D3COLA = D3ADAPTOR(d3).avoidOverlaps(true).size([width_size, height_size]);
const SVG = d3.select(D3NODE).append("svg");
let MAIN_SVG = SVG.append("g").attr("class", "main");

const zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", zoomed);

zoom.filter(function () {
  // Prevent zoom when mouse over node.
  return d3.event.target.tagName.toLowerCase() === "svg";
});

SVG.call(zoom);

// Set the default scale size (adjust the scale value as needed)
const defaultScaleSize = 0.5;
zoom.scaleTo(SVG, defaultScaleSize);

function zoomed() {
  MAIN_SVG.attr("transform", d3.event.transform).call(zoom);
}

function draw_graph(){
    // TODO: For now is not incremental, we draw all the time all the GRAPH!
    MAIN_SVG.selectAll("g").remove();

    const constraints = [];
    const groups = _.groupBy(NODES, "level");

    for (const level of Object.keys(groups)) {
        const node_group = groups[level];
        const constraint = {
            type: "alignment",
            axis: "y",
            offsets: [],
        };
        let prev_node_id = -1;
        for (const node of node_group) {
            constraint.offsets.push({
                node: _.findIndex(NODES, (d) => d.id === node.id),
                offset: 0,
            });

            if (prev_node_id !== -1) {
                constraints.push({
                    axis: "x",
                    left: _.findIndex(NODES, (d) => d.id === prev_node_id),
                    right: _.findIndex(NODES, (d) => d.id === node.id),
                    gap: GAP,
                });
            }

            prev_node_id = node.id;
        }

        constraints.push(constraint);
    }
    SVG
        .attr("viewBox", `0 0 ${width_size} ${height_size}`)
        .style("width", "100%")
        .style("height", "100%");

    D3COLA
        .nodes(NODES)
        .links(EDGES)
        .constraints(constraints)
        .flowLayout("y", 150)
        .linkDistance(50)
        .symmetricDiffLinkLengths(40)
        .avoidOverlaps(true)
        .start(50, 50, 150);


    var link = MAIN_SVG
        .append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(EDGES)
        .enter()
        .append("line")
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y + MARGIN)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y + MARGIN)
        .attr("stroke", "grey")
        .attr("stroke-width", 3);

    var node = MAIN_SVG
        .append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(NODES)
        .enter()
        .append("circle")
        .attr("class", (d) => d.status)
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y + MARGIN)
        .attr("r", DEFAULT_RADIUS)
        .style("cursor", "pointer")
        .call(D3COLA.drag);

    var text = MAIN_SVG
        .append("g")
        .attr("class", "texts")
        .selectAll("text")
        .data(NODES)
        .enter()
        .append("text")
        .attr("font-family", "sans-serif")
        .attr("text-anchor", "middle")
        .attr("font-size", "10px")
        .attr("fill", "white")
        .attr("x", (d) => d.x)
        .attr("y", (d) => d.y + MARGIN)
        .attr("class", (d) => d.status)
        .text((d) => (typeof d.score === "string") ? d.score : parseFloat(d.score.toFixed(2)))
        .style("cursor", "pointer")
        .call(D3COLA.drag);

    node.on("click", (d) => {
        on_show_node(d.id);
        d3.event.stopPropagation();
    });

    text.on("click", (d) => {
        on_show_node(d.id);
        d3.event.stopPropagation();
    });

    SVG.on("click", () => {
        $(".show-menu").html("");
    });

    // node.on({
    //     "mouseover": (d) => {
    //         d3.select(this).style("cursor", "pointer"); 
    //     },
    //     "mouseout": (d) => {
    //         d3.select(this).style("cursor", "default"); 
    //     }
    // });
        
    D3COLA.on("tick", () => {
        link
        .attr("x1", function (d) {
            return d.source.x;
        })
        .attr("y1", function (d) {
            return d.source.y + MARGIN;
        })
        .attr("x2", function (d) {
            return d.target.x;
        })
        .attr("y2", function (d) {
            return d.target.y + MARGIN;
        });

        node
        .attr("cx", function (d) {
            return d.x;
        })
        .attr("cy", function (d) {
            return d.y + MARGIN;
        });

        text
        .attr("x", function (d) {
            return d.x;
        })
        .attr("y", function (d) {
            return d.y + MARGIN;
        });
    });
}