export interface RCNResult {
  /**
   * The output of the neural network in the form of the class index.
   */
  output: number;

  /**
   * The image the neural network used to classify. Stored as base64.
   */
  img: string;
}

export async function getRCNResult(): Promise<RCNResult> {
  return await fetch("http://127.0.0.1:8080/").then((response) =>
    response.json()
  );
}
