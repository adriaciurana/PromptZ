var DEFAULT_PARAMS = {
    "runtime_config": {
        "max_population": 3,
        "topk_population": 1,
        "iterations": 3,
        "generator_samples": 3,
    },
    "llm": "M0",
    "population_creator": {
        "name": "GeneratorPopulationCreator",
        "params": {"num_samples": 3},
    },
    "generator": {
        "name": "LLMSimilarSentencesGenerator",
        "params": {},
    },
    "evaluator": {
        "name": "BERTSimilarityEvaluator",
        "params": {"max_batch": 10},
    },
    "initial_prompt": "Greet me as your friend",
    "target": "Hello my enemy",
};
var WS;
var NODES = [];
var NODES_DICT = {};
var NODE_ID_TO_JSID = {};
var EDGES = [];
var CHROMOSOMES = {};


function start_btn(){
    let params = DEFAULT_PARAMS; // TODO: copy in the future.
    send_cmd("run", params); 
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
            console.log(parent_id)
            EDGES.push({
                source: parseInt(NODE_ID_TO_JSID[parent_id]),
                target: current_jsid,
            });
        }
    }
    draw_graph();
}

function generated_graph(msg_json){
    let iteration = msg_json["iteration"];
    let variations = msg_json["variations"];
    for(let i = 0; i < variations.length; i++){
        let chromosome = variations[i];
        let node = {
            id: NODES.length, 
            level: iteration + 2,
            status: "ok", 
            score: chromosome.score,
        };
        let current_jsid = NODES.length;
        NODE_ID_TO_JSID[chromosome.id] = current_jsid;
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
}

function filtered_graph(msg_json){
    // let iteration = msg_json["iteration"];
    let current_status_population = msg_json["current_status_population"];
    for(let chromosome_id in current_status_population){
        let value = current_status_population[chromosome_id];
        NODES[NODE_ID_TO_JSID[chromosome_id]].status = (value) ? 'ok' : 'killed';
    }
    draw_graph();
}

function results_graph(msg_json){
    // let iteration = msg_json["iteration"];
    let population_ids = msg_json["population_ids"];
    for(let i = 0; i < population_ids.length; i++){
        let chromosome_id = population_ids[i];
        NODES[NODE_ID_TO_JSID[chromosome_id]].status = 'result';
    }
    draw_graph();
}

/* WEB SOCKET */

// Based on https://github.com/cankav/simple_websocket_example

$(document).ready(function () {
    // let params = DEFAULT_PARAMS; // TODO: copy in the future.
    WS = open_ws();
    // setTimeout(()=>{ 
    //     send_cmd("run", params); 
    // }, 2000);
}); // ready end

function open_ws(){
    // websocket on same server with address /websocket
    let ws = new WebSocket("ws://localhost:4003/websocket");
    ws.onopen = function(){
        // Web Socket is connected, send data using send()
        console.log("ws open");
        is_connected = true;
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
const COLORS = {
    "killed": "#A01928",
    "result": "#9F62A4",
    "ok": "#0B7A7A",
};
let MARGIN = 80;
let GAP = 70;

const D3NODE = document.querySelector("#d3Node");

const status2color = (level) => {
    return COLORS[level];
};

// Define cola
const D3ADAPTOR = cola.d3adaptor;
const D3COLA = D3ADAPTOR(d3).avoidOverlaps(true).size([SIZE, SIZE]);
const SVG = d3.select(D3NODE).append("svg");
let MAIN_SVG = SVG.append("g").attr("class", "main");

const zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", zoomed);
zoom.filter(function(){
    // Prevent zoom when mouse over node.
    return d3.event.target.tagName.toLowerCase() === "svg";
});
SVG.call(zoom);
function zoomed(){
    MAIN_SVG.attr("transform", d3.event.transform);
}

function draw_graph(){
    // console.log(NODES);
    // console.log(EDGES);

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
        .attr("viewBox", `0 0 ${SIZE} ${SIZE}`)
        .style("width", "100%")
        .style("height", "auto");

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
        .attr("fill", (d) => status2color(d.status))
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y + MARGIN)
        .attr("r", DEFAULT_RADIUS)
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
        .attr("class", "labelText")
        .text((d) => (typeof d.score === "string") ? d.score : parseFloat(d.score.toFixed(2)))
        .call(D3COLA.drag);
        
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



