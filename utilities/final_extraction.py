from multiprocessing import Pool
from contextlib import closing
import multiprocessing
import pandas as pd
import argparse
import glob
import os


def merge_on_smiles(pred_file):
    print("Merging " + os.path.basename(pred_file) + "...")

    # Read the predictions
    pred = pd.read_csv(pred_file, names=["id", "score"])
    pred.drop_duplicates()

    # Read the smiles
    smile_file = os.path.join(args.smile_dir, os.path.basename(pred_file))
    smi = pd.read_csv(smile_file, delimiter=" ", names=["smile", "id"])
    smi = smi.drop_duplicates()
    return pd.merge(pred, smi, how="inner", on=["id"]).set_index("id")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-smile_dir", required=True, help='Path to SMILES directory for the database')
    parser.add_argument("-prediction_dir", required=True, help='Path to morgan_1024_predicitions of last iteration')
    parser.add_argument("-processors", required=True, help='Number of CPUs for multiprocessing')
    parser.add_argument("-mols_to_dock", required=False, type=int, help='Desired number of molecules to dock')

    args = parser.parse_args()
    predictions = []

    # Find all smile files
    print("Morgan Dir: " + args.prediction_dir)
    print("Smile Dir: " + args.smile_dir)
    for file in glob.glob(args.prediction_dir + "/*"):
        if "smile" in os.path.basename(file):
            print(" - " + os.path.basename(file))
            predictions.append(file)

    # Create a list of pandas dataframes
    print("Finding smiles...")
    print(int(args.processors), len(predictions))
    print("Number of CPUs: " + str(multiprocessing.cpu_count()))
    num_jobs = min(len(predictions), int(args.processors))
    print(num_jobs)
    with closing(Pool(num_jobs)) as pool:
        combined = pool.map(merge_on_smiles, predictions)

    # combine all dataframes
    print("Combining " + str(len(combined)) + " dataframes...")
    base = pd.concat(combined)
    combined = None

    print("Done combining... Sorting!")
    base = base.sort_values(by="score", ascending=False)

    print("Resetting Index...")
    base.reset_index(inplace=True)

    print("Finished Sorting... Here is the base:")
    print(base.head())

    if args.mols_to_dock is not None:
        mtd = args.mols_to_dock
        print("Molecules to dock:", mtd)
        print("Total molecules:", len(base))

        if len(base) <= mtd:
            print("Our total molecules are less or equal than the number of molecules to dock -> saving all molecules")
        else:
            print(f"Our total molecules are more than the number of molecules to dock -> saving {mtd} molecules")
            base = base.head(mtd)

    print("Saving")
    # Rearrange the smiles
    smiles = base.drop('score', 1)
    smiles = smiles[["smile", "id"]]
    print("Here is the smiles:")
    print(smiles.head())
    smiles.to_csv("smiles.csv", sep=" ", index=False)

    # Rearrange for id,score
    base.drop("smile", 1, inplace=True)
    base.to_csv("id_score.csv", index=False)
    print("Here are the ids and scores")
    print(base.head())



