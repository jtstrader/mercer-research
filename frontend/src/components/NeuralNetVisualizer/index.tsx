import { useEffect, useState, useRef } from "react";
import "./index.css";

export interface NeuralNetVisualizerProps {
  selected: number | undefined;
}

/*
 * Create a neural network graph for representing data from RCN.
 */
export function NeuralNetVisualizer(props: NeuralNetVisualizerProps) {
  const [_, setPropState] = useState<NeuralNetVisualizerProps>(props);
  const animating = useRef<boolean>(true);
  const prev = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (animating) {
      setTimeout(() => {
        animating.current = false;
        prev.current = props.selected;
      }, 2000);
    }
  });

  useEffect(() => {
    setPropState(props);
    animating.current = true;
  }, [props]);

  // Circle setup
  const radius = 32;
  const nodeCount = 10;
  const strokeWidth = 4;
  const fontSize = 25;
  const x = radius + strokeWidth;
  const y = radius + strokeWidth;

  // Offset x and y values for the left and right sides
  const xs = (x: number) => x + radius * xShiftValue;
  const ys = (y: number, i: number) => y + radius * yShiftValue * i;

  // Transform left X and right X (XL, XR) for the drawn connections
  const transformXL = (x: number): number => x + radius + strokeWidth;
  const transformXR = (x: number): number => x - radius - strokeWidth;

  // Line setup
  const selectedLineWidth = 2;
  const deselectedLineWidth = selectedLineWidth + 1;

  // Determines the distance between nodes on the x and y axes
  const xShiftValue = 15;
  const yShiftValue = 3;

  // For viewing truncate 6 nodes on the left so the lines can be seen easier
  const leftLayerTrunc = 6;
  const rightLayerTrunc = 0;

  /**
   * Generate left nodes for the neural network graph.
   */
  const leftNodes = (trunc: number = 0): JSX.Element[] => {
    let l: JSX.Element[] = [];
    trunc = Math.floor(trunc / 2);
    for (let i = trunc; i < nodeCount - trunc; i++) {
      l.push(
        <circle
          cx={x}
          cy={ys(y, i)}
          r={radius}
          stroke="white"
          strokeWidth={strokeWidth}
          fill="black"
        />
      );
    }
    return l;
  };

  /**
   * Generate right nodes for the neural network graph.
   */
  const rightNodes = (trunc: number = 0): JSX.Element[] => {
    let l: JSX.Element[] = [];
    for (let i = trunc; i < nodeCount - trunc; i++) {
      l.push(
        <circle
          cx={xs(x)}
          cy={ys(y, i)}
          r={radius}
          stroke="white"
          strokeWidth={strokeWidth}
          fill="black"
        />
      );
    }
    return l;
  };

  /**
   * Generate connections for the neural network graph.
   */
  const grayLines = (trunc: number = 0): JSX.Element[] => {
    let l: JSX.Element[] = [];
    trunc = Math.floor(trunc / 2);

    // Draw `nodeCount` lines from each node to the output node.
    for (let i = trunc; i < nodeCount - trunc; i++) {
      for (let j = 0; j < nodeCount; j++) {
        l.push(
          <line
            className="base-connection"
            x1={transformXL(x)}
            x2={transformXR(xs(x))}
            y1={ys(y, i)}
            y2={ys(y, j)}
            strokeWidth={deselectedLineWidth}
          />
        );
      }
    }

    return l;
  };

  /**
   * Draw the selected line.
   */
  const selectedLine = (trunc: number): JSX.Element[] => {
    let l: JSX.Element[] = [];
    trunc = Math.floor(trunc / 2);

    if (props.selected !== undefined && animating.current) {
      for (let i = trunc; i < nodeCount - trunc; i++) {
        console.log("working...");
        l.push(
          <line
            className="selected-connection animate-node-draw"
            x1={transformXL(x)}
            x2={transformXR(xs(x))}
            y1={ys(y, i)}
            y2={ys(y, props.selected)}
            strokeWidth={selectedLineWidth}
          />
        );
      }
    }
    return l;
  };

  /**
   * Clear a previously selected line.
   */
  const clearLine = (trunc: number): JSX.Element[] => {
    let l: JSX.Element[] = [];
    trunc = Math.floor(trunc / 2);

    if (prev.current !== undefined && animating.current) {
      for (let i = trunc; i < nodeCount - trunc; i++) {
        l.push(
          <line
            className="selected-connection"
            x1={transformXL(x)}
            x2={transformXR(xs(x))}
            y1={ys(y, i)}
            y2={ys(y, prev.current)}
            strokeWidth={selectedLineWidth}
          />
        );
        l.push(
          <line
            className="deselected-connection animate-node-clear"
            x1={transformXL(x)}
            x2={transformXR(xs(x))}
            y1={ys(y, i)}
            y2={ys(y, prev.current)}
            strokeWidth={deselectedLineWidth}
          />
        );
      }
    }
    return l;
  };

  /*
   * Draw the numbers in the respective nodes that can be selected.
   */
  const text = (): JSX.Element[] => {
    let l: JSX.Element[] = [];

    for (let i = 0; i < nodeCount; i++) {
      l.push(
        <text
          x={xs(x) - fontSize / 3.2}
          y={ys(y, i) + fontSize / 3.2}
          fill="white"
          fontSize={fontSize}
        >
          {i}
        </text>
      );
    }
    return l;
  };

  return (
    <svg width={xs(x) + radius + strokeWidth} className="min-h-screen">
      {leftNodes(leftLayerTrunc) /* Truncate 6 nodes on the left */}
      {rightNodes(rightLayerTrunc) /* Truncate nothing on the output layer */}
      {grayLines(leftLayerTrunc) /* Base gray lines for the graph */}
      {clearLine(leftLayerTrunc) /* Clear the previous selection */}
      {selectedLine(leftLayerTrunc) /* Color the selected line orange */}
      {text()}
    </svg>
  );
}

export default NeuralNetVisualizer;
