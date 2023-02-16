import dayjs from "@/utils/dayjs";

export class DateRange {
    private Start: dayjs.Dayjs;
    private End: dayjs.Dayjs;
    private currentLocale: string = "en";

    constructor(start: dayjs.Dayjs, end: dayjs.Dayjs) {
        this.Start = start;
        this.End = end;
    }

    format(): string {
        const localizedStart = this.Start.locale(this.currentLocale);
        const localizedEnd = this.End.locale(this.currentLocale);

        if (this.isSameMonth()) {
            switch (this.currentLocale) {
                case "ko-kr":
                    return `${localizedStart.format("YYYY년 MMM D일")} - ${localizedEnd.format("D일")}`;
                case "ja-jp":
                    return `${localizedStart.format("YYYY年MMMD日")}から${localizedEnd.format("D日まで")}`;
                case "my-mm":
                    return `${localizedStart.format("D")} - ${localizedEnd.format("D ရက် MMMM YYYY")}`;
                case "vi-vn":
                    return `${localizedStart.format("Ngày D")} - ${localizedEnd.format("D [Tháng] MM [Năm] YYYY")}`;
                case "th-th":
                    return `${localizedStart.format("D")} - ${localizedEnd.format("D MMMM BBBB")}`;
                case "zh-cn":
                    return `${localizedStart.format("YYYY年MMMD日")}至${localizedEnd.format("D日")}`;
                default:
                    return `${localizedStart.format("D")} - ${localizedEnd.format("D MMMM YYYY")}`;
            }
        }
        
        if (this.isSameYear()) {
            switch (this.currentLocale) {
                case "ko-kr":
                    return `${localizedStart.format("YYYY년 MMM D일")} - ${localizedEnd.format("MMM D일")}`;
                case "ja-jp":
                    return `${localizedStart.format("YYYY年MMMD日")}から${localizedEnd.format("MMMD日まで")}`;
                case "my-mm":
                    return `${localizedStart.format("D ရက် MMMM")} - ${localizedEnd.format("D ရက် MMMM YYYY")}`;
                case "vi-vn":
                    return `${localizedStart.format("Ngày D [Tháng] MM")} - ${localizedEnd.format("Ngày DD [Tháng] M [Năm] YYYY")}`;
                case "th-th":
                    return `${localizedStart.format("D MMMM")} - ${localizedEnd.format("D MMMM BBBB")}`;
                case "zh-cn":
                    return `${localizedStart.format("YYYY年MMMD日")}至${localizedEnd.format("MMMD日")}`;
                case "id-id":
                    return `${localizedStart.format("D MMMM")} - ${localizedEnd.format("D MMMM YYYY")}`;
                default:
                    return `${localizedStart.format("D MMMM")} - ${localizedEnd.format("D MMMM, YYYY")}`;
            }
        }
    
        switch (this.currentLocale) {
                case "ko-kr":
                    return `${localizedStart.format("YYYY년 MMM D일")} - ${localizedEnd.format("YYYY년 MMM D일")}`;
                case "ja-jp":
                    return `${localizedStart.format("YYYY年MMMD日")}から${localizedEnd.format("YYYY年MMMD日まで")}`;
                case "my-mm":
                    return `${localizedStart.format("D ရက် MMMM YYYY")} - ${localizedEnd.format("D ရက် MMMM YYYY")}`;
                case "vi-vn":
                    return `${localizedStart.format("Ngày D [Tháng] MM [Năm] YYYY")} đến ${localizedEnd.format("Ngày D [Tháng] M [Năm] YYYY")}`;
                case "th-th":
                    return `${localizedStart.format("D MMMM BBBB")} - ${localizedEnd.format("D MMMM BBBB")}`;
                case "zh-cn":
                    return `${localizedStart.format("YYYY年MMMD日")}至${localizedEnd.format("YYYY年MMMD日")}`;
                case "id-id":
                    return `${localizedStart.format("D MMM YYYY")} - ${localizedEnd.format("D MMM YYYY")}`;
                default:
                    return `${localizedStart.format("D MMMM YYYY")} - ${localizedEnd.format("D MMMM YYYY")}`;
            }
    }

    locale(targetLocale: string): DateRange {
        let nextDateRange = new DateRange(this.Start, this.End);
        nextDateRange.currentLocale = targetLocale;
        return nextDateRange;
    }

    private isSameMonth(): boolean {
        return this.Start.isSame(this.End, "month");
    }

    private isSameYear(): boolean {
        return this.Start.isSame(this.End, "year");
    }
}

