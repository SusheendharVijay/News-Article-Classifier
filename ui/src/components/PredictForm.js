import { useState } from "react";
import "./predictForm.modules.css";

const PredictForm = (props) => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  const onSubmitHandler = (event) => {
    event.preventDefault();
    props.onSubmit({
      title: title,
      description: description,
    });
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
            onChange={(e) => setTitle(e.target.value)}
          />
          <br />
          <label for="lname">Description: </label>

          <input
            placeholder="Enter description"
            type="text"
            id="lname"
            name="lname"
            onChange={(e) => setDescription(e.target.value)}
          />
          <button type="submit">Get Prediction</button>
          <button type="button">Retrain Model</button>
        </div>
      </form>
    </div>
  );
};

export default PredictForm;
