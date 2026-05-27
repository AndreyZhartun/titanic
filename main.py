from ml_pipeline.pipeline import MLPipeline


def main():
    pipeline = MLPipeline()
    pipeline.run()
    pipeline.predict()


if __name__ == "__main__":
    main()
