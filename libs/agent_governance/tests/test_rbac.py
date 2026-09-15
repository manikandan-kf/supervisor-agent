from agent_governance.rbac import RbacPolicy


def test_mapped_role_is_allowed():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    assert policy.check("BA", "requirement-agent").allowed


def test_unmapped_agent_is_denied():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    decision = policy.check("BA", "deployment-agent")
    assert not decision.allowed


def test_unknown_role_is_denied():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    assert not policy.check("Intern", "requirement-agent").allowed


def test_missing_role_or_agent_is_denied():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    assert not policy.check("", "requirement-agent").allowed
    assert not policy.check("BA", "").allowed
