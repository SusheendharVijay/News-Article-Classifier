import { useState } from "react";
import "./predictForm.modules.css";

const PredictForm = (props) => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [trainStatus, setTrainStatus] = useState("");

  const onSubmitHandler = (event) => {
    event.preventDefault();
    props.onSubmit({
      title: title,
      description: description,
    });
  };

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  const retrainModel = async () => {
    setTrainStatus("Retraining the model!");
    const response = await fetch("/retrain_model");
    const data = await response.json();
    setTrainStatus("Retraining complete!");
    await sleep(150);
    setTrainStatus("");

    console.log(data);
  };

  return (
    <div className="inter">
      <form onSubmit={onSubmitHandler}>
        <div className="fields">
          <label for="fname">Title: </label>

          <input
            placeholder="Enter the title"
            type="text"
            id="fname"
            name="fname"
            onChange={(e) => {
              props.onReset();
              setTitle(e.target.value);
            }}
          />
          <br />
          <label for="lname">Description: </label>

          <input
            placeholder="Enter description"
            type="text"
            id="lname"
            name="lname"
            onChange={(e) => {
              props.onReset();
              setDescription(e.target.value);
            }}
          />
          <button type="submit">Get Prediction</button>
          <button type="button" onClick={retrainModel}>
            Retrain Model
          </button>
          {trainStatus !== "" && (
            <p style={{ color: "black", width: "200px", fontSize: "20px" }}>
              {trainStatus}
            </p>
          )}
        </div>
      </form>
    </div>
  );
};

export default PredictForm;
