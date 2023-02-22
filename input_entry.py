import argparse
from utils.prediction import *
from utils.regressor import *

# Entry point for prediction from text (model .pth files needed)

ww_models = {
    "gsop": "G-NON-GEO+GEO-ONLY-O1",
    "gmop": "G-NON-GEO+GEO-ONLY-O5",
    "psop": "P-NON-GEO+GEO-ONLY-O1",
    "pmop": "P-NON-GEO+GEO-ONLY-O5"
}

outcomes = 5  # 1 or 5
prob = True
features = ["NON-GEO", "GEO-ONLY"]
text_example = "CIA and FBI can track anyone, and you willingly give the data away"


def main():
    parser = argparse.ArgumentParser(description='Prediction of geolocations')
    parser.add_argument('-o', '--outcomes', type=int, default=outcomes, help="Number of outcomes (lomg, lat) per tweet")
    parser.add_argument('-p', '--prob', action="store_true", help="Use probabilistic model (default: geospatial)")
    parser.add_argument('-m', '--model', type=str, default=None, help='Filename prefix of local model')
    parser.add_argument('-t', '--text', type=str, default=None, help='Text to process (max: 300 words)')
    args = parser.parse_args()

    weighted = outcomes > 1
    covariance = "spher" if prob else None

    if outcomes > 1:
        model_prefix = ww_models["pmop"] if prob else ww_models["gmop"]
    else:
        model_prefix = ww_models["psop"] if prob else ww_models["gsop"]

    prefix = args.model if args.model else model_prefix

    prediction = ModelOutput(BERTregModel(args.outcomes, covariance, weighted, features), prefix)
    print(f"MODEL\tBERT geo regression model is ready, you can now predict location from the text (300 words max) "
          f"in a form of {'Gaussian distributions (lon, lat, cov)' if prob else 'coordinates (lon, lat)'}"
          f" with {outcomes} possible prediction outcomes.\nNOTE\tOutcomes that have very low weight won't be displayed")

    text = args.text if args.text else input("Insert text: ")
    while text != "exit":
        if len(text.split()) < 300:
            prediction.prediction_output(text)

            if outcomes > 1:
                ind = np.argwhere(np.round(prediction.result.weights[0, :] * 100, 2) > 0)
                significant = prediction.result.means[0, ind].reshape(-1, 2)
                sig_weights = prediction.result.weights[0, ind].flatten()
            else:
                significant = prediction.result.means.reshape(-1, 2)
                sig_weights = np.ones(1)

            for i in range(len(sig_weights)):
                weight = np.round(sig_weights[i] * 100, 2)
                point = f"lon: {'  lat: '.join(map(str, significant[i]))}"
                if weight > 0:
                    print(f"\tOut {i + 1}\t{weight}%\t-\t{point}")
        else:
            print(f"Number of words is above 300, unable to process.")

        text = args.text if args.text else input("Insert text: ")


if __name__ == "__main__":
    main()
