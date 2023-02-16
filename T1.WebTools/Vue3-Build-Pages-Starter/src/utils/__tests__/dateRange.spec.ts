import { describe, it } from "vitest";
import { DateRange } from "../dateRange";
import dayjs from "@/utils/dayjs";

describe("DateRange test", () => {
    test.each([
        ["en", "6 - 26 December 2021"],
        ["zh-cn", "2021年12月6日至26日"],
        ["id-id", "6 - 26 Desember 2021"],
        ["th-th", "6 - 26 ธันวาคม 2564"],
        ["vi-vn", "Ngày 6 - 26 Tháng 12 Năm 2021"],
        ["my-mm", "6 - 26 ရက် ဒီဇင်ဘာ 2021"],
        ["ja-jp", "2021年12月6日から26日まで"],
        ["ko-kr", "2021년 12월 6일 - 26일"],
    ])(
        "should handle same month long display",
        (language: string, expectedDisplay: string) => {
            const dateRange = new DateRange(dayjs("2021-12-06"), dayjs("2021-12-26"));
            expect(dateRange.locale(language).format()).toBe(expectedDisplay);
        }
    );

    test.each([
        ["en", "26 November - 3 December, 2021"],
        ["zh-cn", "2021年11月26日至12月3日"],
        ["id-id", "26 November - 3 Desember 2021"],
        ["th-th", "26 พฤศจิกายน - 3 ธันวาคม 2564"],
        ["vi-vn", "Ngày 26 Tháng 11 - Ngày 03 Tháng 12 Năm 2021"],
        ["my-mm", "26 ရက် နိုဝင်ဘာ - 3 ရက် ဒီဇင်ဘာ 2021"],
        ["ja-jp", "2021年11月26日から12月3日まで"],
        ["ko-kr", "2021년 11월 26일 - 12월 3일"],
    ])(
        "should handle same year different month long display",
        (language: string, expectedDisplay: string) => {
            const dateRange = new DateRange(dayjs("2021-11-26"), dayjs("2021-12-03"));
            expect(dateRange.locale(language).format()).toBe(expectedDisplay);
        }
    );

    test.each([
        ["en", "1 December 2021 - 1 January 2022"],
        ["zh-cn", "2021年12月1日至2022年1月1日"],
        ["id-id", "1 Des 2021 - 1 Jan 2022"],
        ["th-th", "1 ธันวาคม 2564 - 1 มกราคม 2565"],
        ["vi-vn", "Ngày 1 Tháng 12 Năm 2021 đến Ngày 1 Tháng 1 Năm 2022"],
        ["my-mm", "1 ရက် ဒီဇင်ဘာ 2021 - 1 ရက် ဇန်နဝါရီ 2022"],
        ["ja-jp", "2021年12月1日から2022年1月1日まで"],
        ["ko-kr", "2021년 12월 1일 - 2022년 1월 1일"],
    ])(
        "should handle different year long display",
        (language: string, expectedDisplay: string) => {
            const dateRange = new DateRange(dayjs("2021-12-01"), dayjs("2022-01-01"));
            expect(dateRange.locale(language).format()).toBe(expectedDisplay);
        }
    );
});
