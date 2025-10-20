using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class LoginWhiteListEntityConfiguration : IEntityTypeConfiguration<LoginWhiteListEntity>
    {
        public void Configure(EntityTypeBuilder<LoginWhiteListEntity> builder)
        {
            builder.ToTable("LoginWhiteList");

            builder.HasKey(x => x.LoginWhiteListId);

            builder.Property(x => x.LoginWhiteListId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.FromDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ToDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

        }
    }
}
