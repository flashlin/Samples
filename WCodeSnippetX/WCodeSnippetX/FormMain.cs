using System.Data;

namespace WCodeSnippetX
{
	public partial class FormMain : Form
	{
		readonly DataTable _table = new DataTable();
		BindingSource _bindingSource = new BindingSource();
		readonly DataGridView _dataGridView = new DataGridView();

		public FormMain()
		{
			InitializeComponent();
			Init();
		}

		void Init()
		{
			_table.Columns.Add(new DataColumn("idx", typeof(int)));
			_table.Columns.Add("content");

			DataRow row = _table.NewRow();
			row[0] = "1";
			row[1] = "Datagridview and richtextbox for bold substring in C#";
			_table.Rows.Add(row);
			_table.AcceptChanges();

			_bindingSource.DataSource = _table;

			_dataGridView.Columns.Add(new DataGridViewComboBoxColumn()
			{
				Name = "index",
				DataPropertyName = "idx",
				DefaultCellStyle = new DataGridViewCellStyle()
				{
					NullValue = 0,
				}
			});
			_dataGridView.Columns.Add(new DataGridViewComboBoxColumn()
			{
				Name = "content",
				DataPropertyName = "content",
				DefaultCellStyle = new DataGridViewCellStyle()
				{
					NullValue = string.Empty,
				}
			});

			_dataGridView.AutoGenerateColumns = false;
			_dataGridView.DataSource = _bindingSource;
			this.Controls.Add(_dataGridView);
		}
	}
}