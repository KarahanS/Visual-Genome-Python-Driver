renderComparison = function(selector, scene_graph) {
    var objects = scene_graph['objects'];
    var attributes = scene_graph['attributes'];
    var relationships = scene_graph['relationships'];

    var obj_type = "object", pred_type = "pred", attr_type = "attr";
    var nodes = [], links = [];
    var object_pred_map = {};

    // these three methods make links and nodes
    function makeLink(parentElementID, childElementID) {
		return new joint.dia.Link({
			source: {id: parentElementID},
			target: {id: childElementID},
			attrs: {'.marker-target': {stroke:'rgba(68,68,68,0.6)',
									   fill: 'rgba(68,68,68,0.6)',
									   d: 'M 4 0 L 0 2 L 4 4 z' },
					'.connection':{stroke: 'rgba(68,68,68,0.6)',
                                   'stroke-width': '1px' }},
			smooth: true,
		});
	  }

    function makeLink2(parentElementID, tx, ty) {
		return new joint.dia.Link({
			source: {id: parentElementID},
			target: {x: tx, y: ty},
			attrs: { '.marker-target': {stroke:'rgba(68,68,68,0.6)',
                                        fill: 'rgba(68,68,68,0.6)',
                                        d: 'M 4 0 L 0 2 L 4 4 z' },
					 '.connection':{stroke: 'rgba(68,68,68,0.6)',
                                    'stroke-width': '1px' }},
			smooth: true,
		});
	  }

	function makeElement(label, indexID, nodeclass, x, y) {
        // print label
        console.log(label);
        x = typeof x !== 'undefined' ? x : 0;
        y = typeof y !== 'undefined' ? y : 0;
        var maxLine = _.max(label.split('\n'), function(l) { return l.length - (l.length - l.replace(/i/g, "").replace(/l/g, "").length); });
        var maxLineLength = maxLine.length - 0.62*(maxLine.length - maxLine.replace(/i/g, "").replace(/l/g,"").length);
        // Compute width/height of the rectangle based on the number
        // of lines in the label and the letter size. 0.6 * letterSize is
        // an approximation of the monospace font letter width.
        var letterSize = 8;
        var width = 5 + (letterSize * (maxLineLength + 1));
        var height = 10 + ((label.split('\n').length + 1) * letterSize);

        return new joint.shapes.basic.Rect({
            id: indexID,
            size: { width: width, height: height },
            attrs: {
                text: { text: label, 'font-size': letterSize },
                rect: {
                    width: width, height: height,
                    rx: 6, ry: 6,
                    stroke: '#555'
                }
            },
            position:{x: x, y: y}
		});
	}

    for(var i = 0; i < objects.length; i++) {
        var node = {
            label: objects[i]['name'],
            class: obj_type
        };
        nodes.push(node);
        object_pred_map[i] = [];
    }
    // Previous predicate finding function remains the same
    function findIndexOfPredicate(arr, predicate){
        for(var i=0; i<arr.length; i++){
            if (arr[i].pred == predicate)
                return i;
        }
        return -1;
    }
    // for each object, predicate, subject, we make a node 
    // (if it doesn't already exist)
    for(var i = 0; i < relationships.length; i++) {
        var t = 0;
        var subject = relationships[i]['subject'];
        var pred = relationships[i]['predicate'];
        var index = findIndexOfPredicate(object_pred_map[subject], pred);
        if (index == -1) {
          var node = {
            label: pred,
            class: pred_type
          };
          nodes.push(node);
          object_pred_map[subject].push({pred: pred, index: nodes.length-1});
          t = nodes.length-1;
        }
        else {
            t = object_pred_map[subject][index].index;
        }
        links.push({
            source : subject,
            target : t,
            weight : 1
        });
        links.push({
            source : t,
            target : relationships[i]['object'],
            weight: 1
        });
    };

    for (var i = 0; i < attributes.length; i++) {
        var unary = {label: attributes[i]['attribute'],
                     class: attr_type};
        nodes.push(unary);
        links.push({source: attributes[i]['object'],
                    target: nodes.length - 1,
                    weight: 1});
    }
    var num_nodes = nodes.length;

    var w = 1200, h = 1000;

    var graph = new joint.dia.Graph();
    var container = document.querySelector(selector);
    if (!container) {
        console.error('Container not found:', selector);
        return;
    }
    var paper = new joint.dia.Paper({
        el: container,
        width: container.offsetWidth || 800,  // Default to 800 if container width is 0
        height: 500,  // Fixed initial height
        gridSize: 1,
        model: graph,
        interactive: false
    });
    // Disable pointer events
    paper.$el.css('pointer-events', 'none');

    V(paper.viewport).translate(20, 20);

    // Create and add all elements
    var elements = [];
    for (var i = 0; i < nodes.length; i++) {
        elements.push(makeElement(nodes[i].label, String(i), nodes[i].class));
    }
    for (var i = 0; i < links.length; i++) {
        elements.push(makeLink(String(links[i].source), String(links[i].target)));
    }
    graph.addCells(elements);

    // Add classes to nodes
    for (var i = 0; i < nodes.length; i++) {
        V(paper.findViewByModel(String(i)).el).addClass(nodes[i].class);
    }

    // Layout the graph
    var graphLayout = joint.layout.DirectedGraph.layout(graph, {
        setLinkVertices: false,
        nodeSep: 10,
        rankSep: 20,
        rankDir: 'LR',
        marginX: 20,
        marginY: 20
    });

    // Adjust paper dimensions to fit content
    paper.setDimensions(graphLayout.width + 40, graphLayout.height + 40);

    // Center the content if smaller than container
    var paperWidth = graphLayout.width + 40;
    var containerWidth = container.offsetWidth;
    if (paperWidth < containerWidth) {
        var dx = (containerWidth - paperWidth) / 2;
        V(paper.viewport).translate(dx, 0);
    }

    return paper;  // Return paper for potential future reference
};