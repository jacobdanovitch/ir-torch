import datasets


def main():
    ds = datasets.load_dataset("jhu-clsp/rank1-R1-MSMARCO", split="train")
    ds = ds.select_columns(["id", "query", "query_id", "title", "text", "answer"])
    print("Loaded dataset with columns:", ds.column_names)

    df = ds.to_pandas()
    df["label"] = (df["answer"].str.lower() == "true").astype(int)
    df["text"] = "# " + df["title"] + "\n" + df["text"]

    df = df.drop(columns=["title", "answer"])
    print(df.head())

    df["items"] = df[["id", "text", "label"]].to_dict(orient="records")
    df = df.groupby(["query_id", "query"]).agg({"items": list})
    df = df.reset_index()

    stats = df.copy()
    stats["num_items"] = stats["items"].apply(len)
    stats["num_pos"] = stats["items"].apply(lambda x: sum(item["label"] for item in x))
    print(stats[["num_items", "num_pos"]].describe())

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_json("/tmp/train.jsonl", orient="records", lines=True)  # noqa: S108
    test_df.to_json("/tmp/test.jsonl", orient="records", lines=True)  # noqa: S108


if __name__ == "__main__":
    main()
