import PredictForm from "./components/PredictForm";
import Card from "./components/Card";
import "./App.css";
function App() {
  return (
    <div className="App">
      <div className="heading">
        <h1>News Article Classifier</h1>
      </div>
      <header className="App-header">
        <Card>
          <PredictForm />
        </Card>
      </header>
    </div>
  );
}

export default App;
