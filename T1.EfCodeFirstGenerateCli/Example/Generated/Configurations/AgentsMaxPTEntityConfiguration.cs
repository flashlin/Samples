using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentsMaxPTEntityConfiguration : IEntityTypeConfiguration<AgentsMaxPTEntity>
    {
        public void Configure(EntityTypeBuilder<AgentsMaxPTEntity> builder)
        {
            builder.ToTable("AgentsMaxPT");

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

            builder.Property(x => x.RoleID)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MaxPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(250)")
                .HasMaxLength(250)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
