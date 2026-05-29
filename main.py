from ml_pipeline.classification.pipeline import ClassificationPipeline


def main():
    pipeline = ClassificationPipeline()

    pipeline.run()
    pipeline.predict()


if __name__ == "__main__":
    main()
