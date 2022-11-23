import argparse
from utils.prediction import *
from utils.regressor import *

# Entry point for prediction from text (model .pth files needed)

ww_models = {
    "gsop": "U-NON-GEO+GEO-ONLY-O1-d-total_type-mf_mean-NP-N30e5-B10-E3-cosine-LR[1e-05;1e-06]",
    "gmop": "U-NON-GEO+GEO-ONLY-O5-d-total_type-mf_mean-NP-weighted-N30e5-B10-E3-cosine-LR[1e-05;1e-06]",
    "psop": "U-NON-GEO+GEO-ONLY-O1-d-total_mean-mf_mean-pos_spher-N30e5-B10-E3-cosine-LR[1e-05;1e-06]",
    "pmop": "U-NON-GEO+GEO-ONLY-O5-d-total_mean-mf_mean-pos_spher-weighted-N30e5-B10-E3-cosine-LR[1e-05;1e-06]"
}

outcomes = 1
prob = False
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
          f" with {outcomes} possible prediction outcomes.\nNOTE\tOutcomes what have very low weight won't be displayed")

    text = args.text if args.text else input("Insert text: ")
    if len(text.split()) < 300:
        prediction.prediction_output(text)
    else:
        print(f"Number of words is above 300, unable to process.")


if __name__ == "__main__":
    main()
