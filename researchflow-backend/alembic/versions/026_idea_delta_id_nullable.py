"""Drop NOT NULL on contribution_to_canonical_idea.idea_delta_id.

The v018 migration absorbed `idea_deltas` into `delta_cards`, and the
concept_synthesizer_service now writes `delta_card_id` instead of
`idea_delta_id`. The legacy NOT NULL constraint blocks every concept
synthesis (synthesize_concepts hits IntegrityError on every paper).

We don't drop the column itself yet — there may be historical rows we want
to keep readable for audit. Just relax the constraint.

Revision ID: 026
Revises: 025
"""

revision = "026"
down_revision = "025"

from alembic import op


def upgrade() -> None:
    op.alter_column("contribution_to_canonical_idea", "idea_delta_id",
                    nullable=True)


def downgrade() -> None:
    # Down only succeeds if you've backfilled idea_delta_id for every row.
    op.alter_column("contribution_to_canonical_idea", "idea_delta_id",
                    nullable=False)
