from ml_pipeline.regression.pipeline import RegressionPipeline


def main():
    pipeline = RegressionPipeline()

    pipeline.run()
    pipeline.show_results()
    pipeline.predict()


if __name__ == "__main__":
    main()
