import argparse
from utils.prediction import *
from utils.regressor import *

# Entry point for prediction from text (model .pth files needed)

local_ww_models = {
    "gsop": "G-NON-GEO+GEO-ONLY-O1",
    "gmop": "G-NON-GEO+GEO-ONLY-O5",
    "psop": "P-NON-GEO+GEO-ONLY-O1",
    "pmop": "P-NON-GEO+GEO-ONLY-O5"
}

outcomes = 5  # 1 or 5
prob = True  # True or False

features = ["NON-GEO", "GEO-ONLY"]

text_example = "CIA and FBI can track anyone, and you willingly give the data away"

local = False
hub_model_prefix = "k4tel/geo-bert-multilingual"


def main():
    parser = argparse.ArgumentParser(description='Prediction of geolocations')
    parser.add_argument('-o', '--outcomes', type=int, default=outcomes, help="Number of outcomes (lomg, lat) per tweet (default: 5)")
    parser.add_argument('-s', '--spat', action="store_true", help="Use geospatial model (default: probabilistic)")
    parser.add_argument('-l', '--local', action="store_true", help="Use model stored locally")
    parser.add_argument('-m', '--model', type=str, default=None, help='Filename prefix of local model OR HuggingFace repository link')
    parser.add_argument('-t', '--text', type=str, default=None, help='Text to process (max: 300 words)')
    args = parser.parse_args()

    weighted = args.outcomes > 1
    covariance = None if args.spat else "spher"

    if args.model:  # models/final/<prefix>.pth file; NOTE correct setup is needed
        prefix = args.model
    elif args.model is None and args.local:  # picking local model according to the setup
        if outcomes > 1:
            local_model_prefix = local_ww_models["gmop"] if args.spat else local_ww_models["pmop"]
        else:
            local_model_prefix = local_ww_models["gsop"] if args.spat else local_ww_models["psop"]

        prefix = local_model_prefix
    else:  # setup for P-NON-GEO+GEO-ONLY-O5
        weighted = True
        covariance = "spher"
        args.outcomes = 5
        args.spat = False
        prefix = hub_model_prefix

    # if not local - loading automatically on BERTregModel init
    model_wrapper = BERTregModel(args.outcomes, covariance, weighted, features, None, prefix) \
        if not args.local else BERTregModel(args.outcomes, covariance, weighted, features)

    # if local - loading automatically on ModelOutput init
    prediction = ModelOutput(model_wrapper, prefix, args.local)

    print(f"MODEL\tBERT geo regression model is ready, you can now predict location from the text (300 words max) "
          f"in a form of {'Gaussian distributions (lon, lat, cov)' if prob else 'coordinates (lon, lat)'}"
          f" with {outcomes} possible prediction outcomes.\nNOTE\tOutcomes that have very low weight won't be displayed")

    text = args.text if args.text else input("Insert text: ")
    while text != "exit":
        if len(text) == 0:
            text = text_example
        if len(text.split()) < 300:
            result = prediction.prediction_output(text, filtering=True, visual=False)

            if args.outcomes > 1:
                ind = np.argwhere(np.round(result.weights[0, :] * 100, 2) > 0)
                significant = result.means[0, ind].reshape(-1, 2)
                weights = result.weights[0, ind].flatten()
            else:
                significant = result.means.reshape(-1, 2)
                weights = np.ones(1)

            sig_weights = np.round(weights * 100, 2)
            sig_weights = sig_weights[sig_weights > 0]

            print(f"RESULT\t{len(sig_weights)} significant prediction outcome(s):")

            for i in range(len(sig_weights)):
                point = f"lon: {'  lat: '.join(map(str, significant[i]))}"
                print(f"\tOut {i + 1}\t{sig_weights[i]}%\t-\t{point}")

        else:
            print(f"Number of words is above 300, unable to process.")

        text = args.text if args.text else input("Insert text: ")


if __name__ == "__main__":
    main()
