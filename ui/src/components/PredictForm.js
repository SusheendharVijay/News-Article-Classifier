import "./predictForm.modules.css";
const PredictForm = () => {
  return (
    <div className="inter">
      <form>
        <div className="fields">
          <label for="fname">Title: </label>

          <input
            placeholder="Enter the title"
            type="text"
            id="fname"
            name="fname"
          />
          <br />
          <label for="lname">Description: </label>

          <input
            placeholder="Enter description"
            type="text"
            id="lname"
            name="lname"
          />
        </div>
      </form>
    </div>
  );
};

export default PredictForm;
