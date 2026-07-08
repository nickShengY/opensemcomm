from opensemcom.cli.download_datasets import build_parser


def test_download_parser_supports_staging_skip_flags():
    args = build_parser().parse_args(["--skip-ag-news", "--skip-bdd-parquet", "--skip-bdd-images"])
    assert args.skip_ag_news is True
    assert args.skip_bdd_parquet is True
    assert args.skip_bdd_images is True


def test_download_parser_defaults_keep_existing_behavior():
    args = build_parser().parse_args([])
    assert args.skip_ag_news is False
    assert args.skip_bdd_parquet is False
    assert args.skip_bdd_images is False
