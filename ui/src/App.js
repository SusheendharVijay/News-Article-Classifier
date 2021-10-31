// import fetch from "fetch";
import axios from "axios";
import { useState, useEffect } from "react";
import PredictForm from "./components/PredictForm";
import Card from "./components/Card";
import "./App.css";
function App() {
  const [payload, setPayload] = useState({ title: "", description: "" });

  const SubmitHandler = (newPayload) => {
    setPayload(newPayload);
  };
  const getPrediction = async () => {
    const Url = "http://localhost:8000/ping";
    // const params = {
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    //   body: JSON.stringify(payload),
    //   method: "post",
    // };

    // const response = await fetch("ping", {
    //   headers: {
    //     "Content-Type": "application/json",
    //     Accept: "application/json",
    //     method: "GET",
    //   },
    // });
    // const data = await response.json();
    // console.log(data);
    // const response = await fetch("/ping");
    // const data = await response.json();
    // console.log(data);

    const response = await axios.get(Url);
  };

  useEffect(() => {
    getPrediction();
  }, [payload]);
  return (
    <div className="App">
      <div className="heading">
        <h1>News Article Classifier</h1>
      </div>
      <header className="App-header">
        <Card>
          <PredictForm onSubmit={SubmitHandler} />
        </Card>

        <p>{}</p>
      </header>
    </div>
  );
}

export default App;
