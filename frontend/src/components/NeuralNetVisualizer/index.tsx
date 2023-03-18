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

  useEffect(() => {
    if (animating) {
      setTimeout(() => {
        animating.current = false;
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

  const xs = (x: number) => x + radius * xShiftValue;
  const ys = (y: number, i: number) => y + radius * yShiftValue * i;

  // Line setup
  const lineWidth = 2;

  // Determines the distance between nodes on the x and y axes.
  const xShiftValue = 15;
  const yShiftValue = 3;

  // For viewing truncate 6 nodes on the left so the lines can be seen easier.
  const leftLayerTrunc = 6;
  const rightLayerTrunc = 0;

  /**
   * Generate left nodes for the neural network graph.
   */
  const leftNodes = (trunc: number = 0): JSX.Element[] => {
    let l: JSX.Element[] = [];
    trunc /= 2;
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
  const lines = (trunc: number = 0): JSX.Element[] => {
    // xl => xl + radius * 2
    // xr => xr - radius * 2
    // y  => y + radius
    const transformXL = (x: number): number => x + radius + strokeWidth;
    const transformXR = (x: number): number => x - radius - strokeWidth;

    let l: JSX.Element[] = [];
    trunc /= 2;

    const genLine = (
      i: number,
      j: number,
      className = "disconnected-node"
    ): JSX.Element => {
      return (
        <line
          className={className}
          x1={transformXL(x)}
          x2={transformXR(xs(x))}
          y1={ys(y, i)}
          y2={ys(y, j)}
          strokeWidth={lineWidth}
        />
      );
    };

    // Draw `nodeCount` lines from each node to the output node.
    for (let i = trunc; i < nodeCount - trunc; i++) {
      for (let j = 0; j < nodeCount; j++) {
        l.push(genLine(i, j));
      }
    }

    // Explictly declare animation, then wrap around a static drawing
    // so the animation can be cancelled safely without a flickering effect.
    if (props.selected && animating.current) {
      for (let i = trunc; i < nodeCount - trunc; i++) {
        l.push(
          genLine(
            i,
            props.selected,
            `connected-node ${animating ? "animate-node-draw" : ""}`
          )
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
    <svg width={xs(x) + radius + strokeWidth}>
      {leftNodes(leftLayerTrunc) /* Truncate 6 nodes on the left */}
      {rightNodes(rightLayerTrunc) /* Truncate nothing on the output layer */}
      {lines(leftLayerTrunc) /* Truncate 6 nodes on the left */}
      {text()}
    </svg>
  );
}

export default NeuralNetVisualizer;
