using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WCodeSnippetX.ViewComponents
{
	public class DataGridViewRichTextBoxColumn : DataGridViewColumn
   {
      public DataGridViewRichTextBoxColumn()
          : base(new DataGridViewRichTextBoxCell())
      {
      }

      public override DataGridViewCell CellTemplate
      {
         get
         {
            return base.CellTemplate;
         }
         set
         {
            if (!(value is DataGridViewRichTextBoxCell))
               throw new InvalidCastException("CellTemplate must be a DataGridViewRichTextBoxCell");

            base.CellTemplate = value;
         }
      }
   }
}
