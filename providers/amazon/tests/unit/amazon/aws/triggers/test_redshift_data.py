# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

from unittest import mock

import pytest

from airflow.providers.amazon.aws.hooks.redshift_data import (
    ABORTED_STATE,
    FAILED_STATE,
    RedshiftDataQueryAbortedError,
    RedshiftDataQueryFailedError,
)
from airflow.providers.amazon.aws.triggers.redshift_data import RedshiftDataTrigger
from airflow.triggers.base import TriggerEvent

TEST_CONN_ID = "aws_default"
TEST_TASK_ID = "123"
POLL_INTERVAL = 4.0


class TestRedshiftDataTrigger:
    def test_redshift_data_trigger_serialization(self):
        """
        Asserts that the RedshiftDataTrigger correctly serializes its arguments
        and classpath.
        """
        trigger = RedshiftDataTrigger(
            statement_id=[],
            task_id=TEST_TASK_ID,
            aws_conn_id=TEST_CONN_ID,
            poll_interval=POLL_INTERVAL,
        )
        classpath, kwargs = trigger.serialize()
        assert classpath == "airflow.providers.amazon.aws.triggers.redshift_data.RedshiftDataTrigger"
        assert kwargs == {
            "statement_id": [],
            "task_id": TEST_TASK_ID,
            "poll_interval": POLL_INTERVAL,
            "aws_conn_id": TEST_CONN_ID,
            "region_name": None,
            "botocore_config": None,
            "verify": None,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("return_value", "response"),
        [
            (
                True,
                TriggerEvent({"status": "success", "statement_id": "uuid"}),
            ),
            (
                False,
                TriggerEvent(
                    {"status": "error", "message": f"{TEST_TASK_ID} failed", "statement_id": "uuid"}
                ),
            ),
        ],
    )
    @mock.patch(
        "airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.check_query_is_finished_async"
    )
    @mock.patch(
        "airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.is_still_running",
        return_value=False,
    )
    async def test_redshift_data_trigger_run(
        self, mocked_is_still_running, mock_check_query_is_finised_async, return_value, response
    ):
        """
        Tests that RedshiftDataTrigger only fires once the query execution reaches a successful state.
        """
        mock_check_query_is_finised_async.return_value = return_value
        trigger = RedshiftDataTrigger(
            statement_id="uuid",
            task_id=TEST_TASK_ID,
            poll_interval=POLL_INTERVAL,
            aws_conn_id=TEST_CONN_ID,
        )
        generator = trigger.run()
        actual = await generator.asend(None)
        assert response == actual

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raised_exception", "expected_response"),
        [
            (
                RedshiftDataQueryFailedError("Failed"),
                {
                    "status": "error",
                    "statement_id": "uuid",
                    "message": "Failed",
                    "type": FAILED_STATE,
                },
            ),
            (
                RedshiftDataQueryAbortedError("Aborted"),
                {
                    "status": "error",
                    "statement_id": "uuid",
                    "message": "Aborted",
                    "type": ABORTED_STATE,
                },
            ),
            (
                Exception(f"{TEST_TASK_ID} failed"),
                {"status": "error", "statement_id": "uuid", "message": f"{TEST_TASK_ID} failed"},
            ),
        ],
    )
    @mock.patch(
        "airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.check_query_is_finished_async"
    )
    @mock.patch(
        "airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.is_still_running",
        return_value=False,
    )
    async def test_redshift_data_trigger_exception(
        self, mocked_is_still_running, mock_check_query_is_finised_async, raised_exception, expected_response
    ):
        """
        Test that RedshiftDataTrigger fires the correct event in case of an error.
        """
        mock_check_query_is_finised_async.side_effect = raised_exception

        trigger = RedshiftDataTrigger(
            statement_id="uuid",
            task_id=TEST_TASK_ID,
            poll_interval=POLL_INTERVAL,
            aws_conn_id=TEST_CONN_ID,
        )
        task = [i async for i in trigger.run()]
        assert len(task) == 1
        assert TriggerEvent(expected_response) in task

    @pytest.mark.asyncio
    @mock.patch("asyncio.sleep")
    @mock.patch(
        "airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.check_query_is_finished_async",
        return_value=True,
    )
    @mock.patch(
        "airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.is_still_running",
        side_effect=[True, False],
    )
    async def test_redshift_data_trigger_run_multiple_polls(
        self,
        mocked_is_still_running,
        mock_check_query_is_finished_async,
        mock_sleep,
    ):
        """
        Tests that RedshiftDataTrigger correctly handles multiple poll cycles
        where is_still_running returns True multiple times before returning False.

        This test verifies the regression gap created by PR #66157's control flow rewrite.
        The previous while await form made multi-poll behavior hard to test; the new
        while True + if-not-break form needs explicit testing to catch silent future regressions.
        """
        trigger = RedshiftDataTrigger(
            statement_id="uuid",
            task_id=TEST_TASK_ID,
            poll_interval=POLL_INTERVAL,
            aws_conn_id=TEST_CONN_ID,
        )

        generator = trigger.run()
        response = await generator.asend(None)

        # Verify is_still_running was called twice: first True (continue polling), then False (break)
        assert mocked_is_still_running.call_count == 2

        # Verify asyncio.sleep was called once (between the two polls)
        assert mock_sleep.call_count == 1

        # Verify a single TriggerEvent was emitted with success status
        assert response == TriggerEvent(
            {"status": "success", "statement_id": "uuid"}
        )
