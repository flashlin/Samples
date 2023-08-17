import { describe, it, expect } from 'vitest'
//import { flushPromises, mount, VueWrapper } from '@vue/test-utils'
import { fetchAllTable, fetchTableData } from "../tableFetcher"

import data_table_json_expected from "./data/datatable_expected.json";
import { readFileContent } from './testHelper';
describe('fetchTableData', () => {
    //const thead_tr_td_table = fs.readFileSync(thead_tr_td_table_FilePath, 'utf-8');
    const thead_tr_td_table = readFileContent("data/thead_tr_td.xml");
    
    //let app: VueWrapper
    //beforeEach(() => {
    // app = mount(LoginName, {
    //     global: {
    //         mocks: {
    //             $t: vi.fn(),
    //         },
    //     },
    // })
    //})

    it('should show tips on first time', async () => {
        //const loginNameInput = app.findByTestId('loginName')
        //await loginNameInput.setValue('test')
        //app.findByTestId('loginNameTips')
        //await flushPromises()
        //expect(getTipListElement().exists()).toBeTruthy()
    })

    it('table thead tr td', () => {
        const fakeTable = document.createElement('table');
        fakeTable.innerHTML = thead_tr_td_table;
        const dataTable = fetchTableData(fakeTable);
        const json = JSON.stringify(dataTable)
        expect(json).toBe(data_table_json_expected)
    })


    it('table', async () => {
        document.body.innerHTML = thead_tr_td_table;
        const dataTableList = fetchAllTable();
        console.log(dataTableList);
    })

})
