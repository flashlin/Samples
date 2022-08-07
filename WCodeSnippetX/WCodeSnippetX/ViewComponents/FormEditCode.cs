using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WCodeSnippetX.Models;

namespace WCodeSnippetX.ViewComponents
{
	public partial class FormEditCode : Form
	{
		private CodeSnippetEntity _codeSnippet;
		private ICodeSnippetRepo _repo;

		public FormEditCode(ICodeSnippetRepo repo)
		{
			_repo = repo;
			InitializeComponent();
		}

		public void SetValue(CodeSnippetEntity codeSnippet)
		{
			_codeSnippet = codeSnippet;
			textBoxProgramLanguage.Text = codeSnippet.ProgramLanguage;
			textBoxContent.Text = codeSnippet.Content;
			textBoxDescription.Text = codeSnippet.Description;
		}

		public void SaveValue()
		{
			_codeSnippet.ProgramLanguage = textBoxProgramLanguage.Text;
			_codeSnippet.Content = textBoxContent.Text;
			_codeSnippet.Description = textBoxDescription.Text;

			if (_codeSnippet.Id != 0)
			{
				_repo.UpdateCode(_codeSnippet);
			}
			else
			{
				_repo.AddCode(_codeSnippet);
			}
		}

		private void buttonSave_Click(object sender, EventArgs e)
		{
			SaveValue();
			Hide();
		}

		private void buttonDelete_Click(object sender, EventArgs e)
		{
			var dialogResult = MessageBox.Show("Delete sure", "Delete", MessageBoxButtons.YesNo);
			if (dialogResult != DialogResult.Yes)
			{
				return;
			}
			if (_codeSnippet.Id == 0)
			{
				return;
			}
			_repo.DeleteCode(_codeSnippet);
			Hide();
		}
	}
}
