from .utils import create_feedback_agent, generate_feedback_csv
from pathlib import Path


def main():
    agent = create_feedback_agent()

    base_path = Path(__file__).parent
    input_csv = base_path / "file.csv"
    output_csv = base_path / "output.csv"

    generate_feedback_csv(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        agent=agent,
        name_column="name",
        experience_column="positive experience",
        feedback_column="feedback",
    )

    print(f"Arquivo gerado: {output_csv}")


if __name__ == "__main__":
    main()
