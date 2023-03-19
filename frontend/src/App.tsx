import { useEffect, useState } from "react";
import "./App.css";

import { RCNResult, getRCNResult } from "./requests/rcn";
import { NeuralNetVisualizer } from "./components/NeuralNetVisualizer";

function App() {
  const [res, setRes] = useState<RCNResult | undefined>(undefined);
  const waitTime = 5000; // Amount of time to wait getting a new response

  useEffect(() => {
    const resInterval = setInterval(
      async () => setRes(await getRCNResult()),
      waitTime
    );

    return () => {
      clearInterval(resInterval);
    };
  }, []);

  return (
    <>
      <div className="text-center text-white font-bold text-5xl p-2">
        Rust Convolutional Network (RCN)
      </div>

      <div
        id="main-display"
        className="min-h-screen w-screen flex columns-2 gap-x-32 justify-center items-center p-5"
      >
        <img
          className={"rcn-image"}
          src={`data:image/png;base64,${res?.img}`}
          alt={""}
        />
        <NeuralNetVisualizer selected={res?.output} />
      </div>
    </>
  );
}

export default App;
