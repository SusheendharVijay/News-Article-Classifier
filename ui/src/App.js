import { useState, useEffect, useCallback } from "react";
import PredictForm from "./components/PredictForm";
import Card from "./components/Card";
import "./App.css";
function App() {
  const [payload, setPayload] = useState({ title: "", description: "" });
  const [prediction, setPrediction] = useState("");

  const SubmitHandler = (newPayload) => {
    setPayload(newPayload);
  };
  const getPrediction = useCallback(async () => {
    const params = {
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      method: "post",
    };

    const response = await fetch("/predict_category", params);
    const data = await response.json();
    if (payload.title !== "" && payload.description !== "") {
      setPrediction(data.news_category);
    }
  }, [payload]);

  // const resetPrediction = () => {
  //   setPrediction("");
  // };

  useEffect(() => {
    getPrediction();
  }, [payload, getPrediction]);
  return (
    <div className="App">
      <div className="heading">
        <h1>News Article Classifier</h1>
      </div>
      <header className="App-header">
        <Card>
          <PredictForm onSubmit={SubmitHandler} onReset={setPrediction} />
        </Card>
        <div style={{ margin: "20px" }}></div>
        <Card>
          <p style={{ color: "black", width: "200px", fontSize: "20px" }}>
            Prediction : {prediction}
          </p>
        </Card>
      </header>
    </div>
  );
}

export default App;
