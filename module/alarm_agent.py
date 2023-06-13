from time import process_time
from typing_extensions import override

import streamlit as st
from .alarm_filter_logic import isStranger, isDanger

WINDOWS_SIZE_SEC = 5


class AlarmAgent:
    def __init__(self, group):
        time = process_time()
        self.timer_start = time
        self.timer_end = time
        self.group: str = group
        self.container_callfunc = None
        self.initial = True

    def start_timer(self):
        self.timer_start = process_time()

    def end_timer(self):
        self.timer_end = process_time()

    def reset_timers(self):
        time = process_time()
        self.timer_start = time
        self.timer_end = time

    def bind_container_callfunc(self, container_func):
        self.container_callfunc = container_func

    def run(self, group_this_frame, boxes):
        if group_this_frame != self.group:
            return
        filtered_count = self._filtered_count(boxes)
        if filtered_count > 0:
            self.start_timer()
        else:
            self.end_timer()

        elapsed = self.timer_end - self.timer_start
        #print(f"⏲️⏲️⏲️: {elapsed} | {'❌' if self.is_alarm_canceled(elapsed) else '✅'}")
        self.on_alarm_canceled() if self.is_alarm_canceled(
            elapsed
        ) else self.on_alarm_persistent(filtered_count)

    def is_alarm_canceled(self, elapsed):
        # elapsed 小于0: 检测到危险
        # elapsed 0~WINDOWS_SIZE_SEC: 没有检测到危险，准备取消报警
        # elapsed 大于WINDOWS_SIZE_SEC: 取消报警
        if elapsed > WINDOWS_SIZE_SEC:
            return True
        if 0 <= elapsed <= WINDOWS_SIZE_SEC:
            if self.initial:
                return True
            return False
        if elapsed < 0:
            self.initial = False
            return False

    def on_alarm_canceled(self):
        pass

    def on_alarm_persistent(self, filtered_count):
        pass

    def _filtered_count(self, boxes):
        pass


class FireAlarmAgent(AlarmAgent):
    @override
    def on_alarm_canceled(self):
        if not self.container_callfunc:
            return
        with self.container_callfunc():
            st.info("未检测到烟雾或火灾。")

    @override
    def on_alarm_persistent(self, filtered_count):
        if not self.container_callfunc:
            return
        with self.container_callfunc():
            st.warning(f"检测到{filtered_count}处可能有烟雾或火灾。", icon="🔥")

    @override
    def _filtered_count(self, boxes):
        return isDanger(boxes)


class FaceAlarmAgent(AlarmAgent):
    @override
    def on_alarm_canceled(self):
        if not self.container_callfunc:
            return
        with self.container_callfunc():
            st.info("未检测到陌生人。")

    @override
    def on_alarm_persistent(self, filtered_count):
        if not self.container_callfunc:
            return
        with self.container_callfunc():
            st.warning(f"检测到{filtered_count}处可能有陌生人员。", icon="👤")

    @override
    def _filtered_count(self, boxes):
        return isStranger(boxes)
