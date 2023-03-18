import { useEffect, useState } from "react";
import "./App.css";

import { RCNResult, getRCNResult } from "./requests/rcn";
import { NeuralNetVisualizer } from "./components/NeuralNetVisualizer";

function App() {
  const [res, setRes] = useState<RCNResult | undefined>(undefined);
  const waitTime = 5000; // Amount of time to wait getting a new response

  useEffect(() => {
    (async () => {
      return new Promise<void>(async (resolve) => {
        setRes(await getRCNResult());
        setTimeout(resolve, waitTime);
      });
    })();
  }, [res]);

  return (
    <>
      <div className="text-center text-white font-bold text-5xl p-2">
        Rust Convolutional Network (RCN)
      </div>

      <div
        id="main-display"
        className="min-h-screen w-screen justify-center flex p-5"
      >
        <NeuralNetVisualizer selected={res?.output} />
      </div>
    </>
  );
}

export default App;
