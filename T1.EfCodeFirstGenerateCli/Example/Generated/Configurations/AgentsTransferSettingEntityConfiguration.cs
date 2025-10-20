using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentsTransferSettingEntityConfiguration : IEntityTypeConfiguration<AgentsTransferSettingEntity>
    {
        public void Configure(EntityTypeBuilder<AgentsTransferSettingEntity> builder)
        {
            builder.ToTable("AgentsTransferSetting");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.TransferSetting)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ParentID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
