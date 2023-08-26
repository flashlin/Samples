/////////////////////////////////////////////////

export interface ITableNameAlia {
    tableName: string;
    tableAlia: string;
}
/**
 * 獲取 sql 中所有的 TableName 和 Alia
 * @param {*} sqlText SQL字符串
 * @return {Array<ITableNameAlia>} []
 */
export const splitTableNameAndAlia = (sqlText: string): Array<ITableNameAlia> => {
    const regTableAliaFrom = /(^|(\s+))from\s+([^\s]+(\s+|(\s+as\s+))[^\s]+(\s+|,)\s*)+(\s+(where|left|right|full|join|inner|union))?/gi;

    const regTableAliaJoin = /(^|(\s+))join\s+([^\s]+)\s+(as\s+)?([^\s]+)\s+on/gi;

    const regTableAliaFromList = sqlText.match(regTableAliaFrom) || [];

    const regTableAliaJoinList = sqlText.match(regTableAliaJoin) || [];

    const strList = [
        ...regTableAliaFromList.map((item) => item
            .replace(/(^|(\s+))from\s+/gi, '')
            .replace(/\s+(where|left|right|full|join|inner|union)((\s+.*?$)|$)/gi, '')
            .replace(/\s+as\s+/gi, ' ')
            .trim()
        ),
        ...regTableAliaJoinList.map((item) => item
            .replace(/(^|(\s+))join\s+/gi, '')
            .replace(/\s+on((\s+.*?$)|$)/, '')
            .replace(/\s+as\s+/gi, ' ')
            .trim()
        ),
    ];

    const tableList: Array<{
        tableName: string;
        tableAlia: string;
    }> = [];

    strList.map((tableAndAlia) => {
        tableAndAlia.split(',').forEach((item) => {
            const tableName = item.trim().split(/\s+/)[0];
            const tableAlia = item.trim().split(/\s+/)[1];
            tableList.push({
                tableName,
                tableAlia,
            });
        });
    });
    return tableList;
};
