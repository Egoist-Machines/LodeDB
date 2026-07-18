"""`lodedb cloud init --agents` artifact generators: content correctness and the
leave-user-files-alone rules (no server needed — pure filesystem)."""

import json

from lodedb.cloud._agents_scaffold import (
    AGENTS_SECTION_HEADING,
    MCP_SERVER_NAME,
    scaffold_agent_artifacts,
)

LINK = dict(host="https://cloud.example.com", org="acme", environment="support", store="docs")


def test_fresh_repo_gets_all_three_artifacts(tmp_path):
    written, notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert written == [".claude/skills/orecloud/SKILL.md", "AGENTS.md", ".mcp.json"]
    assert notes == []

    skill = (tmp_path / ".claude/skills/orecloud/SKILL.md").read_text()
    # Agent Skills spec: frontmatter with name + description (<=1024 chars).
    assert skill.startswith("---\nname: orecloud\ndescription: ")
    description = skill.split("description: ", 1)[1].split("\n", 1)[0]
    assert len(description) <= 1024
    assert skill.count("\n") <= 500
    # Generated from the REAL link, not placeholders.
    assert "acme/support/docs" in skill
    assert "https://cloud.example.com" in skill
    assert "min_seq" in skill  # the async-write rule agents trip on

    agents = (tmp_path / "AGENTS.md").read_text()
    assert agents.startswith(AGENTS_SECTION_HEADING)
    assert "lodedb cloud sync" in agents

    config = json.loads((tmp_path / ".mcp.json").read_text())
    server = config["mcpServers"][MCP_SERVER_NAME]
    assert server["url"] == "https://cloud.example.com/mcp/control"
    # Env expansion, never a literal secret.
    assert server["headers"]["Authorization"] == "Bearer ${ORECLOUD_TOKEN}"


def test_existing_agents_md_gains_section_once(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# My repo\n\nRules.\n")
    written, notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert "AGENTS.md" in written
    content = (tmp_path / "AGENTS.md").read_text()
    assert content.startswith("# My repo")
    assert AGENTS_SECTION_HEADING in content

    # Second run: the section is the repo's to edit now — never regenerated.
    (tmp_path / "AGENTS.md").write_text(content.replace("lodedb cloud sync", "MY EDIT"))
    written, notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert "AGENTS.md" not in written
    assert any("AGENTS.md" in note for note in notes)
    assert "MY EDIT" in (tmp_path / "AGENTS.md").read_text()


def test_existing_mcp_json_is_merged_not_replaced(tmp_path):
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"type": "stdio", "command": "x"}}})
    )
    written, _notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert ".mcp.json" in written
    config = json.loads((tmp_path / ".mcp.json").read_text())
    assert set(config["mcpServers"]) == {"other", MCP_SERVER_NAME}

    # An existing orecloud-control entry is theirs (maybe a different host).
    config["mcpServers"][MCP_SERVER_NAME]["url"] = "https://elsewhere/mcp/control"
    (tmp_path / ".mcp.json").write_text(json.dumps(config))
    written, notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert ".mcp.json" not in written
    assert any(MCP_SERVER_NAME in note for note in notes)
    reread = json.loads((tmp_path / ".mcp.json").read_text())
    assert reread["mcpServers"][MCP_SERVER_NAME]["url"] == "https://elsewhere/mcp/control"


def test_unparseable_mcp_json_is_left_alone(tmp_path):
    (tmp_path / ".mcp.json").write_text("{not json")
    written, notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert ".mcp.json" not in written
    assert any("not valid JSON" in note for note in notes)
    assert (tmp_path / ".mcp.json").read_text() == "{not json"


def test_skill_is_regenerated_each_run(tmp_path):
    scaffold_agent_artifacts(tmp_path, **LINK)
    skill_path = tmp_path / ".claude/skills/orecloud/SKILL.md"
    skill_path.write_text("stale")
    written, _notes = scaffold_agent_artifacts(tmp_path, **LINK)
    assert ".claude/skills/orecloud/SKILL.md" in written
    assert "acme/support/docs" in skill_path.read_text()
