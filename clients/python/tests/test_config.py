import pytest

from ate_env.config import FleetConfig, ValidationError


def test_unknown_fields_are_rejected():
    # Previously silently dropped by pydantic (e.g. the NeMo/TML doc snippets).
    with pytest.raises(ValidationError, match="warm_pool_size"):
        FleetConfig(warm_pool_size=64)
    with pytest.raises(ValidationError, match="warmpool_replicas"):
        FleetConfig.from_dict({"backend": "mock", "warmpool_replicas": 4})


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0},
    {"max_concurrent": 0},
    {"max_warmpool_replicas": -1},
    {"acquire_timeout_s": 0},
    {"backend": "kubernetes"},  # only substrate and mock supported
    {"tenancy": "Bad_Name"},
    {"tenancy": "-leading-dash"},
    {"strategy": "rolling"},
    {"data_plane": "invalid_plane"},
    {"grpc_endpoint": "{actor}.svc:50051"},  # unknown field
])
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        FleetConfig(**kwargs)


def test_template_and_image_templates_configured():
    cfg = FleetConfig(template="my-tmpl", image_templates={"python:3.11": "tmpl-python"})
    assert cfg.template == "my-tmpl"
    assert cfg.image_templates["python:3.11"] == "tmpl-python"


def test_auth_token_is_secret_and_round_trips():
    cfg = FleetConfig(auth_token="s3cr3t")
    assert "s3cr3t" not in repr(cfg)
    assert "s3cr3t" not in str(cfg.model_dump())
    # model_dump() -> from_dict() is how configs are shipped to Ray workers.
    again = FleetConfig.from_dict(cfg.model_dump())
    assert again.auth_token.get_secret_value() == "s3cr3t"
