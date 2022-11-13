import argparse
from utils.prediction import *
from utils.regressor import *

ww_models = {
    "gsop": "U-NON-GEO+GEO-ONLY-O1-d-total_type-mf_mean-NP-N30e5-B10-E3-cosine-LR[1e-05;1e-06]",
    "gmop": "U-NON-GEO+GEO-ONLY-O5-d-total_type-mf_mean-NP-weighted-N30e5-B10-E3-cosine-LR[1e-05;1e-06]",
    "psop": "U-NON-GEO+GEO-ONLY-O1-d-total_mean-mf_mean-pos_spher-N30e5-B10-E3-cosine-LR[1e-05;1e-06]",
    "pmop": "U-NON-GEO+GEO-ONLY-O5-d-total_mean-mf_mean-pos_spher-weighted-N30e5-B10-E3-cosine-LR[1e-05;1e-06]"
}

outcomes = 5
prob = True
features = ["NON-GEO", "GEO-ONLY"]


def main():
    parser = argparse.ArgumentParser(description='Prediction of geolocations')
    parser.add_argument('-o', '--outcomes', type=int, default=outcomes, help="Number of outcomes (lomg, lat) per tweet")
    parser.add_argument('-p', '--prob', action="store_false", help="Use probabilistic model (default: geospatial)")
    args = parser.parse_args()

    weighted = outcomes > 1
    covariance = "spher" if prob else None

    if outcomes > 1:
        model_prefix = ww_models["pmop"] if prob else ww_models["gmop"]
    else:
        model_prefix = ww_models["psop"] if prob else ww_models["gsop"]

    prediction = ModelOutput(BERTregModel(args.outcomes, covariance, weighted, features), model_prefix)
    print(f"MODEL\tBERT geo regression model is ready, you can now predict location from the text (300 words max) "
          f"in a form of {'Gaussian distributions (lon, lat, cov)' if prob else 'coordinates (lon, lat)'}"
          f" with {outcomes} possible prediction outcomes.\nNOTE\tOutcomes what have very low weight won't be displayed")
    text = input("Insert text: ")
    prediction.prediction_output(text)


if __name__ == "__main__":
    main()
